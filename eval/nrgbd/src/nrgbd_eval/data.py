from pathlib import Path
import hashlib
import json
import re
import numpy as np
from .contracts import ModelSceneInput, SceneInput

SCENES = (
    "breakfast_room",
    "complete_kitchen",
    "green_room",
    "grey_white_room",
    "kitchen",
    "morning_apartment",
    "staircase",
    "thin_geometry",
    "whiteroom",
)
INTRINSICS = np.array(
    [[554.2562584220408, 0, 320], [0, 554.2562584220408, 240], [0, 0, 1]],
    dtype=np.float64,
)
PATTERNS = {
    "images": re.compile(r"img(\d+)\.png$"),
    "depth": re.compile(r"depth(\d+)\.png$"),
}


def _ids(directory, kind):
    if not directory.is_dir():
        raise ValueError(f"missing {kind} directory: {directory}")
    out = {}
    for p in directory.iterdir():
        m = PATTERNS[kind].fullmatch(p.name)
        if m:
            key = str(int(m.group(1)))
            if key in out:
                raise ValueError(f"duplicate {kind} frame id {key}")
            out[key] = p.resolve()
    return out


def _poses(path):
    if not path.is_file():
        raise ValueError(f"missing poses: {path}")
    lines = path.read_text().splitlines()
    if len(lines) % 4:
        raise ValueError(f"truncated pose matrices: {path}")
    out = []
    for i in range(0, len(lines), 4):
        try:
            a = np.array(
                [[float(x) for x in row.split()] for row in lines[i : i + 4]],
                dtype=np.float64,
            )
        except ValueError as e:
            raise ValueError(f"invalid pose at frame {i//4}") from e
        if a.shape != (4, 4) or not np.isfinite(a).all():
            raise ValueError(f"nonfinite or malformed pose at frame {i//4}")
        a = a.copy()
        a[:, 1:3] *= -1
        out.append(a)
    return np.stack(out) if out else np.empty((0, 4, 4))


def load_scene(root, scene_id, kf=10):
    root = Path(root).resolve()
    if scene_id not in SCENES:
        raise ValueError(f"unknown NRGBD scene: {scene_id}")
    d = root / scene_id
    rgb = _ids(d / "images", "images")
    dep = _ids(d / "depth", "depth")
    poses = _poses(d / "poses.txt")
    rids = set(rgb)
    dids = set(dep)
    if rids != dids:
        raise ValueError(
            f"modality mismatch in {scene_id}: rgb_only={sorted(rids-dids,key=int)[:5]} depth_only={sorted(dids-rids,key=int)[:5]}"
        )
    ordered = sorted(rids, key=int)
    if any(int(x) >= len(poses) for x in ordered):
        raise ValueError(f"pose count does not cover frame ids in {scene_id}")
    selected = ordered[::kf]
    if not selected:
        raise ValueError(f"empty selection in {scene_id}")
    from PIL import Image

    transforms = []
    intrinsics = []
    for x in selected:
        with Image.open(dep[x]) as im:
            w, h = im.size
        rw = 518
        rh = round(h * (rw / w) / 14) * 14
        if rh != 392:
            raise ValueError(
                f"FastVGGT NRGBD protocol expects 518x392 after preprocessing, got {rw}x{rh} for {dep[x]}"
            )
        left = 0
        top = 0
        K = INTRINSICS.copy()
        K[0, :] *= rw / w
        K[1, :] *= rh / h
        transforms.append((rw, rh, left, top))
        intrinsics.append(K)
    return SceneInput(
        ModelSceneInput(scene_id, tuple(selected), tuple(rgb[x] for x in selected)),
        tuple(dep[x] for x in selected),
        np.stack([poses[int(x)] for x in selected]),
        np.stack(intrinsics),
        tuple(transforms),
    )


def preflight_dataset(root, kf=10):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root missing: {root}")
    present = sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name != "archives"
    )
    missing = sorted(set(SCENES) - set(present))
    extra = sorted(set(present) - set(SCENES))
    if missing or extra:
        raise ValueError(f"scene mismatch: missing={missing} extra={extra}")
    scenes = [load_scene(root, s, kf) for s in SCENES]
    from PIL import Image

    manifest = []
    for scene in scenes:
        pose_path = root / scene.model.scene_id / "poses.txt"
        manifest.append(
            (
                str(pose_path.relative_to(root)),
                hashlib.sha256(pose_path.read_bytes()).hexdigest(),
            )
        )
        for rgb, depth in zip(scene.model.rgb_paths, scene.depth_paths):
            with Image.open(rgb) as image:
                image.verify()
            with Image.open(depth) as image:
                image.verify()
            for path in (rgb, depth):
                stat = path.stat()
                manifest.append(
                    (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)
                )
    payload = {
        "protocol": "fastvggt_nrgbd_kf10_v1",
        "dataset_root": str(root),
        "scenes": list(SCENES),
        "selected_counts": {s.model.scene_id: len(s.model.frame_ids) for s in scenes},
        "frame_ids": {s.model.scene_id: list(s.model.frame_ids) for s in scenes},
    }
    payload["input_fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    payload["ready"] = True
    return payload
