"""Prepare an isolated, checksummed test split without modifying raw 7-Scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
from registered_data import DEFAULT_DATA_ROOT, PROTOCOL, validate_registered_root
from registration import DEPTH_TO_RGB, register_depth

REFERENCE = "https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py"


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def _prepare_frame(task):
    source, destination, sequence, frame = task
    source = Path(source) / sequence
    destination = Path(destination) / sequence
    destination.mkdir(parents=True, exist_ok=True)
    source_depth = source / f"frame-{frame}.depth.png"
    raw_bytes = source_depth.read_bytes()
    raw = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.shape != (480, 640) or raw.dtype != np.uint16:
        raise ValueError(f"Expected raw 640x480 uint16 depth: {source_depth}")
    registered = register_depth(raw)
    ok, encoded = cv2.imencode(".png", registered, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"PNG encoding failed: {source_depth}")
    if not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED), registered):
        raise IOError(f"PNG round-trip changed registered depth: {source_depth}")
    payload = encoded.tobytes()
    target = destination / f"frame-{frame}.depth.proj.png"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    temporary = target.with_suffix(f".png.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, target)
    for suffix in ("color.png", "pose.txt"):
        (destination / f"frame-{frame}.{suffix}").symlink_to(source / f"frame-{frame}.{suffix}")
    return {"sequence": sequence, "frame": frame,
            "source_depth": str(source_depth),
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "registered_sha256": hashlib.sha256(payload).hexdigest(),
            "registered_bytes": len(payload),
            "valid_pixels": int(((registered > 0) & (registered <= 10000)).sum()),
            "changed_pixels": int((registered != raw).sum())}


def _worker_init():
    cv2.setNumThreads(1)


def prepare_dataset(source_root, output_root, *, workers=8):
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).absolute()
    resolved_output = output_root.resolve()
    if source_root == resolved_output or source_root in resolved_output.parents or resolved_output in source_root.parents:
        raise ValueError("Source and output trees must be separate")
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"Output must be a new directory: {output_root}")
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = []
    splits = []
    sequences = set()
    for split in sorted(source_root.glob("*/TestSplit.txt")):
        splits.append(split)
        for value in split.read_text().splitlines():
            number = int("".join(filter(str.isdigit, value)))
            sequence = f"{split.parent.name}/seq-{number:02}"
            if sequence in sequences:
                raise ValueError(f"Duplicate test sequence: {sequence}")
            sequences.add(sequence)
            frames = sorted((source_root / sequence).glob("frame-*.color.png"))
            if not frames:
                raise ValueError(f"Empty test sequence: {sequence}")
            for i, rgb in enumerate(frames):
                frame = f"{i:06}"
                if rgb.name != f"frame-{frame}.color.png":
                    raise ValueError(f"Non-contiguous input frames: {rgb}")
                for suffix in ("depth.png", "pose.txt"):
                    if not (rgb.parent / f"frame-{frame}.{suffix}").is_file():
                        raise FileNotFoundError(f"Missing raw frame triplet: {rgb}")
                tasks.append((str(source_root), str(output_root), sequence, frame))
    if not tasks:
        raise ValueError("No test frames found in the raw source")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Conservative maximum: one uncompressed uint16 PNG plus filesystem metadata.
    required = len(tasks) * 700000 + 1024 ** 3
    if shutil.disk_usage(output_root.parent).free < required:
        raise OSError(f"Need at least {required} free bytes for safe preparation")
    output_root.mkdir()
    metadata = {"protocol": PROTOCOL, "complete": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source_root), "split": "test",
                "frame_count": len(tasks), "sequence_count": len(sequences),
                "reference": REFERENCE, "calibration": "SimpleRecon benchmark approximation",
                "depth_focal": 585.0, "rgb_focal": 525.0,
                "principal_point": [320.0, 240.0], "depth_to_rgb": DEPTH_TO_RGB.tolist(),
                "invalid_raw_depth": [0, 65535], "depth_unit": "millimeter",
                "source_pixel_center_offset": 0.5, "rounding": "nearest-even",
                "projection_source_sha256": hashlib.sha256(
                    Path(__file__).with_name("registration.py").read_bytes()).hexdigest()}
    _write_json(output_root / "registration.json", metadata)
    for split in splits:
        target = output_root / split.parent.name / "TestSplit.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(split.read_bytes())
    _worker_init()
    started = time.monotonic()
    total_bytes = 0
    changed_frames = 0
    empty_frames = 0
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) if workers > 1 else None
    try:
        results = pool.map(_prepare_frame, tasks, chunksize=8) if pool else map(_prepare_frame, tasks)
        with (output_root / "frames.jsonl").open("x") as manifest:
            for count, row in enumerate(results, 1):
                manifest.write(json.dumps(row, separators=(",", ":")) + "\n")
                total_bytes += row["registered_bytes"]
                changed_frames += row["changed_pixels"] > 0
                empty_frames += row["valid_pixels"] == 0
                if count % 250 == 0 or count == len(tasks):
                    manifest.flush()
                    print(json.dumps({"completed": count, "total": len(tasks),
                                      "seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    metadata.update({"complete": True, "registered_bytes": total_bytes,
                     "changed_frames": changed_frames, "empty_valid_frames": empty_frames,
                     "frames_manifest_sha256": hashlib.sha256(
                         (output_root / "frames.jsonl").read_bytes()).hexdigest(),
                     "completed_at": datetime.now(timezone.utc).isoformat()})
    _write_json(output_root / "registration.json", metadata)
    try:
        validate_registered_root(output_root, verify_hashes=True)
    except Exception:
        metadata["complete"] = False
        _write_json(output_root / "registration.json", metadata)
        raise
    print(json.dumps({"complete": True, "root": str(output_root),
                      "frames": len(tasks), "registered_bytes": total_bytes}), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/data/yjh/share/datasets/7scenes")
    parser.add_argument("--output-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify", action="store_true", help="Validate existing output only")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(validate_registered_root(args.output_root, verify_hashes=True), indent=2))
    else:
        prepare_dataset(args.source_root, args.output_root, workers=args.workers)
