"""Apply narrow, backed-up edits to the H20 evaluation readers and launchers.

Run with the feedforwardreconstruct directory as the sole argument. Stops if
the expected old text changed; it never overwrites a concurrently edited file.
"""
from datetime import datetime, timezone
import difflib
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one edit location, found {text.count(old)}: {old!r}")
    return text.replace(old, new, 1)


def planned_edits(root):
    edits = {}

    def edit(relative, replacements):
        path = root / relative
        original = path.read_bytes()
        updated = original.decode()
        for old, new in replacements:
            updated = replace_once(updated, old, new)
        edits[path] = (original, updated.encode())

    fast_helper = ('from pathlib import Path\n'
                   'sys.path.append(str(Path(__file__).resolve().parents[3] / "adapters"))\n')
    edit('eval/7scenes/reference/FastVGGT-main/eval/data.py', [
        ('import os\n', 'import os\nimport sys\n' + fast_helper +
         'from registered_data import registered_depth_path\n'),
        ('            depthpath_proj = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")\n'
         '            depthpath_raw = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.png")\n'
         '            depthpath = depthpath_proj if osp.exists(depthpath_proj) else depthpath_raw\n',
         '            depthpath = registered_depth_path(self.ROOT, scene_id, im_idx)\n'),
    ])
    edit('vggtstream/src/eval/mv_recon/data.py', [
        ('import os\n', 'import os\nimport sys\nfrom pathlib import Path\n'
         'sys.path.append(str(Path(__file__).resolve().parents[4] / "eval" / "7scenes" / "adapters"))\n'
         'from registered_data import registered_depth_path\n'),
        ('            depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")\n'
         '            if not osp.exists(depthpath):\n'
         '                depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.png")\n',
         '            depthpath = registered_depth_path(self.ROOT, scene_id, im_idx)\n'),
    ])
    edit('eval/7scenes/reference/FastVGGT-main/eval/eval_7andN.py', [
        ('import sys\n', 'import sys\n' + fast_helper +
         'from registered_data import DEFAULT_DATA_ROOT, prepare_evaluation\n'),
        ('default="/data/yjh/share/datasets/7scenes", help="7-Scenes dataset root"',
         'default=DEFAULT_DATA_ROOT, help="registered 7-Scenes dataset root"'),
        ('def main(args):\n', 'def main(args):\n'
         '    prepare_evaluation(args.data_root, args.output_dir, kf=args.kf)\n'),
    ])
    edit('eval/7scenes/adapters/stream_launch_7scenes.py', [
        ('import sys\n', 'import sys\nfrom registered_data import DEFAULT_DATA_ROOT, prepare_distributed_evaluation\n'),
        ('default="/data/yjh/share/datasets/7scenes")', 'default=DEFAULT_DATA_ROOT)'),
        ('    accelerator = Accelerator()\n', ''),
        ('def main(args):\n', 'def main(args):\n'
         '    accelerator = Accelerator()\n'
         '    prepare_distributed_evaluation(accelerator, args.data_root, args.output_dir, kf=args.kf)\n'),
    ])
    edit('eval/7scenes/adapters/eval_omega_7scenes.py', [
        ('import argparse\n', 'import argparse\nfrom registered_data import DEFAULT_DATA_ROOT, prepare_evaluation\n'),
        ('default="/data/sy/7scenes")', 'default=DEFAULT_DATA_ROOT)'),
        ('    args = parser.parse_args()\n', '    args = parser.parse_args()\n'
         '    prepare_evaluation(args.data_root, args.output_dir, kf=args.kf)\n'),
    ])
    edit('eval/7scenes/adapters/eval_long_7scenes.py', [
        ('import argparse\n', 'import argparse\nfrom registered_data import DEFAULT_DATA_ROOT, prepare_evaluation\n'),
        ('default="/data/yjh/share/datasets/7scenes")', 'default=DEFAULT_DATA_ROOT)'),
        ('def run(args):\n', 'def run(args):\n'
         '    prepare_evaluation(args.data_root, args.output_dir, kf=args.kf)\n'),
    ])
    for script, prefix in [('run_vggt_job.sh', 'vggt'),
                           ('run_original_vggt_job.sh', 'vggt_original'),
                           ('run_fastvggt_job.sh', 'fastvggt'),
                           ('run_streamvggt_job.sh', 'streamvggt')]:
        replacements = [
            ('KF="$2"\n', 'KF="$2"\n'
             'RUN_TAG="registered_v1_$(date -u +%Y%m%dT%H%M%SZ)_$$"\n'
             'DATA_ROOT="${SEVENSCENES_DATA_ROOT:-/data/yjh/share/datasets/7scenes_registered_simplerecon_v1}"\n'),
            (f'OUT_DIR="$EVAL_DIR/results/{prefix}_kf${{KF}}"',
             f'OUT_DIR="$EVAL_DIR/results/{prefix}_kf${{KF}}_${{RUN_TAG}}"'),
            (f'LOG_FILE="$LOG_DIR/{prefix}_kf${{KF}}.log"',
             f'LOG_FILE="$LOG_DIR/{prefix}_kf${{KF}}_${{RUN_TAG}}.log"'),
            (f'PID_FILE="$LOG_DIR/{prefix}_kf${{KF}}.pid"',
             f'PID_FILE="$LOG_DIR/{prefix}_kf${{KF}}_${{RUN_TAG}}.pid"'),
            (f'VRAM_FILE="$LOG_DIR/vram_{prefix}_kf${{KF}}.csv"',
             f'VRAM_FILE="$LOG_DIR/vram_{prefix}_kf${{KF}}_${{RUN_TAG}}.csv"'),
            ('echo "GPU=$GPU KF=$KF CKPT=$CKPT"',
             'echo "GPU=$GPU KF=$KF CKPT=$CKPT DATA_ROOT=$DATA_ROOT OUT_DIR=$OUT_DIR"'),
        ]
        if prefix == 'streamvggt':
            replacements.append(('--data_root /data/yjh/share/datasets/7scenes', '--data_root "$DATA_ROOT"'))
        else:
            replacements.append(('  --output_dir "$OUT_DIR" \\\n',
                                 '  --output_dir "$OUT_DIR" \\\n  --data_root "$DATA_ROOT" \\\n'))
        edit('eval/7scenes/' + script, replacements)
    return edits


def main():
    root = Path(sys.argv[1]).resolve()
    if not (root / 'vggtstream/src/eval/mv_recon/data.py').is_file():
        raise ValueError('Expected the feedforwardreconstruct project root')
    edits = planned_edits(root)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{os.getpid()}'
    backup = root / 'eval/7scenes/preprocessing/backups' / tag
    backup.mkdir(parents=True)
    patches = []
    for path, (original, updated) in edits.items():
        relative = path.relative_to(root)
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Retain the exact bytes the patch was planned against.
        destination.write_bytes(original)
        if path.read_bytes() != original:
            raise RuntimeError(f'Concurrent edit detected; stopped before changing {path}')
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.registration.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
        shutil.copymode(path, temporary)
        if path.read_bytes() != original:
            temporary.unlink()
            raise RuntimeError(f'Concurrent edit detected; stopped before changing {path}')
        os.replace(temporary, path)
        patches.extend(difflib.unified_diff(original.decode().splitlines(True), updated.decode().splitlines(True),
                                           fromfile=str(relative), tofile=str(relative)))
        print(relative, hashlib.sha256(updated).hexdigest())
    (backup / 'changes.patch').write_text(''.join(patches))
    print('Original files and patch:', backup)


if __name__ == '__main__':
    main()
