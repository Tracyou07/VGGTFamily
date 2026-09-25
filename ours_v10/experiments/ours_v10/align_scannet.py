"""Validate a frozen v10 ScanNet prediction set, then run unchanged v9 sparse stitching."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

from experiments.ours_v6.runtime import sha256, write_json
from experiments.ours_v10.compare_frozen_sparse import execute as align_frozen
from experiments.ours_v10.predict_scannet import (
    CHECKPOINT_SHA256, FROZEN_INPUT, FROZEN_MANIFEST, INPUT_SHA256, MODES,
    expected_windows, validate_frozen_input,
)


def validate_prediction_set(run_root, mode):
    prediction_root = Path(run_root) / mode
    manifest_path = prediction_root / "run_manifest.json"
    if not (prediction_root / "COMPLETE.json").is_file():
        raise ValueError("prediction set is not COMPLETE")
    manifest = json.loads(manifest_path.read_text())
    frames = manifest["configuration"]["frames"]
    ids = [f"{i:06d}" for i in range(frames)]
    windows = expected_windows(frames)
    if (manifest.get("dataset"), manifest.get("scene"), manifest.get("communication_mode")) != (
            "ScanNet", "scene0000_00", mode):
        raise ValueError("not the requested ScanNet prediction set")
    if manifest["frame_ids"] != ids or manifest["windows"] != [list(w) for w in windows]:
        raise ValueError("frozen frame/window order changed")
    if (manifest["input_sha256"] != INPUT_SHA256 or
            manifest["checkpoint_sha256"] != CHECKPOINT_SHA256 or
            Path(manifest["configuration"]["input"]).resolve() != FROZEN_INPUT.resolve()):
        raise ValueError("frozen input/checkpoint identity changed")
    old_source = json.loads(FROZEN_MANIFEST.read_text())
    _, _, _, prefix_hash, _ = validate_frozen_input(FROZEN_INPUT, old_source, frames)
    if prefix_hash != manifest["image_tensor_sha256"]:
        raise ValueError("input prefix tensor hash changed")
    records = manifest["prediction_files"]
    if len(records) != len(windows):
        raise ValueError("incomplete prediction window set")
    for index, (record, (lo, hi)) in enumerate(zip(records, windows)):
        path = prediction_root / "windows" / f"{index:04d}" / "local.npz"
        if (record["window"], record["lo"], record["hi"]) != (index, lo, hi):
            raise ValueError("prediction window metadata mismatch")
        if Path(record["path"]).resolve() != path.resolve() or sha256(path) != record["sha256"]:
            raise ValueError(f"prediction file identity changed: {path}")
    return manifest_path, prediction_root / "windows"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        manifest_path, predictions = validate_prediction_set(args.run_root, args.mode)
        align_frozen(SimpleNamespace(predictions=predictions, source_manifest=manifest_path,
            prediction_set=args.mode, mode="sparse_point_camera_joint",
            output=args.output, vkitti_raw_root=None, vkitti_condition=None))
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "FAILED.json", dict(reason=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
