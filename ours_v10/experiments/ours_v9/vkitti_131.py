"""Virtual KITTI 1.3.1 Scene20 adapter using the existing evaluation package."""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

from experiments.ours_v6.windows import make_windows

EVAL_SOURCE = Path("/home/ubuntu/yjh/feedforwardreconstruct/eval/virtual_kitti/src")
DEFAULT_RAW = Path("/data/yjh/share/datasets/Virtual_KITTI_1.3.1/extracted")
CONDITIONS = {"clone": "Clone", "rain": "Rain", "fog": "Fog"}


def _official():
    if str(EVAL_SOURCE) not in sys.path:
        sys.path.insert(0, str(EVAL_SOURCE))
    from virtual_kitti_eval.config import VirtualKittiConfig
    from virtual_kitti_eval.data import inspect_raw_sequence
    from virtual_kitti_eval.metrics import ate_rmse_m, apply_sim3
    return VirtualKittiConfig, inspect_raw_sequence, ate_rmse_m, apply_sim3


@dataclass(frozen=True)
class ConditionInventory:
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    gt_c2w: np.ndarray
    sources: tuple[dict, ...]


def inspect_condition(raw_root=DEFAULT_RAW, condition="clone", *,
                      expected_frames=837, require_full=True):
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported Scene20 condition: {condition}")
    config_class, inspect, _, _ = _official()
    raw_root = Path(raw_root).resolve()
    sequence = f"Scene20/{CONDITIONS[condition]}"
    # These are validation-only paths. No dataset preparation is performed.
    config = config_class(1, raw_root, raw_root.parent / "archives",
                          Path("/tmp/ours_v9_vkitti_validation_only"),
                          raw_root.parent / "unused_sequences.txt", (sequence,))
    status = inspect(config, sequence)
    if not status.ready:
        raise ValueError("Virtual KITTI 1.3.1 validation failed: " +
                         "; ".join(f"{b.code}: {b.message}" for b in status.blockers))
    raw = status._raw
    if require_full and (len(raw.frame_ids) != 837 or expected_frames != 837):
        raise ValueError("full Scene20 requires exactly 837 frames")
    if not 1 <= expected_frames <= len(raw.frame_ids):
        raise ValueError("invalid requested frame count")
    return ConditionInventory(raw.frame_ids[:expected_frames],
                              raw.image_paths[:expected_frames],
                              raw.poses_c2w[:expected_frames], raw.sources)


def scene20_windows(frame_ids, window_size=60, overlap=10):
    ids = tuple(str(frame) for frame in frame_ids)
    if ids != tuple(f"{i:05d}" for i in range(len(ids))):
        raise ValueError("Scene20 requires contiguous five-digit IDs from 00000")
    windows = make_windows(len(ids), window_size, overlap)
    if len(ids) == 837 and (window_size, overlap) == (60, 10):
        if len(windows) != 17 or windows[-1] != (800, 837):
            raise ValueError("unexpected 837-frame window schedule")
    return windows


def evaluate_trajectory(global_result, raw_root, condition, output,
                        *, expected_frames=837, require_full=True):
    """Call only after stitcher.finish; GT never enters edge fitting."""
    inventory = inspect_condition(raw_root, condition,
                                  expected_frames=expected_frames,
                                  require_full=require_full)
    ids = tuple(str(value) for value in global_result["frame_ids"])
    if ids != inventory.frame_ids:
        raise ValueError("stitched trajectory frame IDs differ from official RGB/GT order")
    poses = np.asarray(global_result["c2w"])
    owner = np.asarray(global_result["source_window"])
    if owner.shape != (len(ids),):
        raise ValueError("invalid ownership vector")
    _, _, ate_fn, apply_sim3 = _official()
    official = ate_fn(ids, poses, inventory.frame_ids, inventory.gt_c2w)
    sim3 = official.alignment
    aligned = apply_sim3(sim3, poses[:, :3, 3])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "evaluation.npz", gt_c2w=inventory.gt_c2w,
                        pred_raw=poses, aligned_centers=aligned,
                        scale=sim3.scale, rotation=sim3.rotation,
                        translation=sim3.translation)
    def angle(rotation):
        return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))
    adjacent = []
    for index in range(1, len(ids)):
        pred = np.linalg.inv(poses[index-1]) @ poses[index]
        target = np.linalg.inv(inventory.gt_c2w[index-1]) @ inventory.gt_c2w[index]
        adjacent.append(dict(before=ids[index-1], after=ids[index],
            boundary=bool(owner[index] != owner[index-1]),
            pred_translation_raw=float(np.linalg.norm(pred[:3, 3])),
            pred_translation_scaled=float(sim3.scale * np.linalg.norm(pred[:3, 3])),
            gt_translation=float(np.linalg.norm(target[:3, 3])),
            pred_rotation_deg=angle(pred[:3, :3]), gt_rotation_deg=angle(target[:3, :3]),
            translation_error=float(np.linalg.norm(sim3.scale*pred[:3, 3]-target[:3, 3])),
            rotation_error_deg=angle(target[:3, :3].T @ pred[:3, :3])))
    boundaries = [row for row in adjacent if row["boundary"]]
    for filename, value in (("adjacent_pose_errors.json", adjacent),
                            ("boundary_diagnostics.json", boundaries),
                            ("within_window_diagnostics.json", [r for r in adjacent if not r["boundary"]])):
        (output / filename).write_text(json.dumps(value, indent=2))
    metrics = dict(protocol_id=official.protocol_id,
                   matched_frames=official.matched_frames,
                   ate_rmse_m=official.rmse_m,
                   alignment=dict(scale=sim3.scale, rotation=sim3.rotation.tolist(),
                                  translation=sim3.translation.tolist()),
                   ownership_boundaries=dict(count=len(boundaries)))
    (output / "trajectory_metrics.json").write_text(json.dumps(metrics, indent=2))
    from experiments.ours_v6.metrics import summarize
    summary = summarize(output)
    return dict(metrics, **{key: value for key, value in summary.items()
                            if key not in metrics})
