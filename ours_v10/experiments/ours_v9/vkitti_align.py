"""CPU-only 2x2 alignment of frozen Virtual KITTI 1.3.1 predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.ours_v6.windows import make_windows
from experiments.ours_v6.runtime import sha256, write_json
from experiments.ours_v9.vkitti_131 import DEFAULT_RAW, scene20_windows

MODES = ("independent", "overlap_correspondence")
ALIGNMENTS = ("point_camera_joint", "sparse_point_camera_joint")
SAME_CONFIG = ("input", "frames", "window_size", "overlap", "backend_profile",
               "correspondence_attention_path", "query_chunk_size",
               "cache_local_kv_dtype", "dense_head_frame_chunk",
               "reuse_image_encoding")


def validate_prediction_pair(left, right):
    if (left.get("communication_mode"), right.get("communication_mode")) != MODES:
        raise ValueError("prediction modes must be independent and overlap_correspondence")
    for label, manifest in zip(MODES, (left, right)):
        ids = manifest["frame_ids"]
        config = manifest["configuration"]
        expected = scene20_windows(ids, config["window_size"], config["overlap"])
        if [list(pair) for pair in manifest["windows"]] != [list(pair) for pair in expected]:
            raise ValueError(f"{label} window schedule disagrees with frame count")
        if len(ids) != config["frames"]:
            raise ValueError(f"{label} frame count disagrees with manifest")
    for key in ("dataset", "scene", "condition", "frame_ids",
                "input_sha256", "image_tensor_sha256", "checkpoint_sha256", "precision"):
        if left.get(key) != right.get(key):
            raise ValueError(f"prediction mismatch: {key}")
    if left["dataset"] != "Virtual KITTI 1.3.1" or left["scene"] != "Scene20":
        raise ValueError("expected Virtual KITTI 1.3.1 Scene20 predictions")
    for key in SAME_CONFIG:
        if left["configuration"].get(key) != right["configuration"].get(key):
            raise ValueError(f"prediction configuration mismatch: {key}")
    if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    return len(left["frame_ids"]), len(left["windows"])


def execute(args):
    root = args.prediction_root
    pair = []
    for mode in MODES:
        folder = root / mode
        if not (folder / "COMPLETE.json").is_file():
            raise ValueError(f"incomplete frozen prediction set: {folder}")
        manifest = json.loads((folder / "run_manifest.json").read_text())
        if manifest.get("condition") != args.condition:
            raise ValueError("condition and prediction manifest disagree")
        pair.append(manifest)
    count, n_windows = validate_prediction_pair(*pair)
    if not args.smoke and (count, n_windows) != (837, 17):
        raise ValueError("full Scene20 requires 837 frames and 17 windows")
    source_input = Path(pair[0]["configuration"]["input"])
    if sha256(source_input) != pair[0]["input_sha256"]:
        raise ValueError("frozen input file hash changed")
    for mode, manifest in zip(MODES, pair):
        paths = sorted((root / mode / "windows").glob("*/local.npz"))
        if len(paths) != n_windows:
            raise ValueError(f"{mode} has incomplete local.npz windows")
        for i, (path, expected) in enumerate(zip(paths, manifest["prediction_files"])):
            if path.name != "local.npz" or path.parent.name != f"{i:04d}":
                raise ValueError("window prediction order changed")
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"frozen prediction changed: {path}")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    write_json(args.output / "source_manifest.json", dict(
        dataset="Virtual KITTI 1.3.1", scene="Scene20", condition=args.condition,
        prediction_root=str(root), source_manifests=[str(root/m/"run_manifest.json") for m in MODES],
        frame_count=count, window_count=n_windows, edge_count=n_windows-1,
        input_sha256=pair[0]["input_sha256"], checkpoint_sha256=pair[0]["checkpoint_sha256"],
        gt_used_only_after_stitching=True))
    results = []
    for mode in MODES:
        for align in ALIGNMENTS:
            output = args.output / f"{mode}_{align}"
            command = [sys.executable, "-B", "-u", "-m",
                "experiments.ours_v9.compare_frozen_sparse",
                "--predictions", str(root/mode/"windows"),
                "--source-manifest", str(root/mode/"run_manifest.json"),
                "--prediction-set", mode, "--mode", align, "--output", str(output),
                "--vkitti-raw-root", str(args.raw_root),
                "--vkitti-condition", args.condition]
            with (args.output / f"{mode}_{align}.log").open("x") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f"CPU alignment failed: {mode}/{align}; see {output}")
            results.append(json.loads((output / "alignment_comparison.json").read_text()))
    write_json(args.output / "comparison.json", dict(
        status="success", results=results, frame_count=count,
        edge_count=n_windows-1))
    write_json(args.output / "COMPLETE.json", dict(status="complete"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(("clone", "rain", "fog")), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--smoke", action="store_true", help="allow explicitly shortened input")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        execute(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "FAILED.json", dict(reason=str(error),
            traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
