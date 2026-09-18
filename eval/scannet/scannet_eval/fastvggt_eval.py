"""Strict adapter for the reused FastVGGT ScanNet evaluator.

Protocol ``fastvggt_scannet_evo132`` evaluates normalized GT as world-to-camera poses while
independently aligning predicted geometry by bounding-box center and diagonal.
This module preserves those conventions and adds validation around the reused
implementation; it does not reinterpret the scores as camera-center Sim(3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .vendor.fastvggt_eval_utils import evaluate_scene_and_save


PROTOCOL_ID = "fastvggt_scannet_evo132"


_METRIC_KEYS = (
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
    "scale_factor",
    "aligned_chamfer_distance",
    "aligned_ate",
    "aligned_are",
    "aligned_rpe_rot",
    "aligned_rpe_trans",
    "aligned_scale_factor",
)


class FastVGGTEvaluationError(ValueError):
    """Raised when an input or reused FastVGGT result is not a valid evaluation."""


def _frame_ids(value: Any, *, owner: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) < 3:
        raise FastVGGTEvaluationError(f"{owner} must provide at least 3 frame_ids as a tuple")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) or item < 0 for item in value):
        raise FastVGGTEvaluationError(f"{owner} frame_ids must be non-negative integers")
    normalized = tuple(int(item) for item in value)
    if tuple(sorted(set(normalized))) != normalized:
        raise FastVGGTEvaluationError(f"{owner} frame_ids must be unique and strictly increasing")
    return normalized


def _rigid_poses(value: Any, *, owner: str, count: int) -> np.ndarray:
    try:
        poses = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FastVGGTEvaluationError(f"{owner} poses_c2w must be a numeric array") from error
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise FastVGGTEvaluationError(
            f"{owner} poses_c2w must have shape ({count}, 4, 4) with finite values"
        )
    if not np.allclose(poses[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise FastVGGTEvaluationError(f"{owner} poses_c2w must be homogeneous rigid transforms")
    rotations = poses[:, :3, :3]
    identity = np.eye(3)[None]
    if not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, identity, atol=1e-5):
        raise FastVGGTEvaluationError(f"{owner} poses_c2w must contain rigid rotations")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5):
        raise FastVGGTEvaluationError(f"{owner} poses_c2w must contain proper rigid rotations")
    return poses


def _require_evo_trajectory_rank(poses_w2c: np.ndarray, *, owner: str) -> None:
    positions = poses_w2c[:, :3, 3]
    centered = positions - positions.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered))
    if rank < 2:
        raise FastVGGTEvaluationError(
            f"{owner} camera trajectory is degenerate: evo w2c rank must be at least 2; got {rank}"
        )


def _decode_gt_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FastVGGTEvaluationError(f"missing GT PLY: {path}")
    try:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
    except Exception as error:  # Open3D exposes format failures through several exception types.
        raise FastVGGTEvaluationError(f"GT PLY failed to decode: {path}") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise FastVGGTEvaluationError(f"GT PLY decoded to an empty point cloud: {path}")
    if not np.isfinite(points).all():
        raise FastVGGTEvaluationError(f"GT PLY points must be finite: {path}")
    return points


def validate_scene_inputs(scene: Any) -> None:
    """Validate a SceneData-like object, including fully decoding its GT PLY."""

    scene_id = getattr(scene, "scene_id", None)
    if not isinstance(scene_id, str) or not scene_id:
        raise FastVGGTEvaluationError("scene_id must be a non-empty string")
    frame_ids = _frame_ids(getattr(scene, "frame_ids", None), owner="scene")
    image_paths = getattr(scene, "image_paths", None)
    if not isinstance(image_paths, tuple) or len(image_paths) != len(frame_ids):
        raise FastVGGTEvaluationError("scene image_paths must match frame_ids")
    poses = _rigid_poses(getattr(scene, "poses_c2w", None), owner="scene", count=len(frame_ids))
    normalized_c2w = np.linalg.inv(poses[0]) @ poses
    _require_evo_trajectory_rank(np.linalg.inv(normalized_c2w), owner="scene")
    try:
        gt_ply = Path(getattr(scene, "gt_ply"))
    except (TypeError, AttributeError) as error:
        raise FastVGGTEvaluationError("scene must provide a GT PLY path") from error
    _decode_gt_points(gt_ply)


def _validate_prediction(scene: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray, float]:
    scene_ids = _frame_ids(scene.frame_ids, owner="scene")
    prediction_ids = _frame_ids(getattr(prediction, "frame_ids", None), owner="prediction")
    if prediction_ids != scene_ids:
        raise FastVGGTEvaluationError(
            f"prediction frame_ids do not match scene frame_ids: {prediction_ids} != {scene_ids}"
        )
    try:
        points = np.asarray(getattr(prediction, "points"), dtype=np.float64)
    except (TypeError, ValueError, AttributeError) as error:
        raise FastVGGTEvaluationError("prediction points must be a numeric array") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise FastVGGTEvaluationError("prediction points must be a non-empty array with shape (M, 3)")
    if not np.isfinite(points).all():
        raise FastVGGTEvaluationError("prediction points must be finite")
    poses = _rigid_poses(
        getattr(prediction, "poses_c2w", None), owner="prediction", count=len(prediction_ids)
    )
    _require_evo_trajectory_rank(np.linalg.inv(poses), owner="prediction")
    raw_seconds = getattr(prediction, "inference_seconds", None)
    if isinstance(raw_seconds, bool):
        raise FastVGGTEvaluationError("prediction inference_seconds must be finite and non-negative")
    try:
        inference_seconds = float(raw_seconds)
    except (TypeError, ValueError) as error:
        raise FastVGGTEvaluationError(
            "prediction inference_seconds must be finite and non-negative"
        ) from error
    if not np.isfinite(inference_seconds) or inference_seconds < 0:
        raise FastVGGTEvaluationError("prediction inference_seconds must be finite and non-negative")
    return points, poses, inference_seconds


def _validated_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise FastVGGTEvaluationError("FastVGGT evaluator returned no metrics")
    missing = [key for key in _METRIC_KEYS if key not in metrics]
    if missing:
        raise FastVGGTEvaluationError(f"FastVGGT evaluator returned missing metrics: {missing}")
    result: dict[str, float] = {}
    for key in _METRIC_KEYS:
        try:
            value = float(metrics[key])
        except (TypeError, ValueError) as error:
            raise FastVGGTEvaluationError(f"FastVGGT metric {key} is not numeric") from error
        if not np.isfinite(value):
            raise FastVGGTEvaluationError(f"FastVGGT metrics must be finite; {key}={value}")
        result[key] = value
    return result


def evaluate_prediction(
    scene: Any,
    prediction: Any,
    output_dir: str | Path,
    *,
    chamfer_max_dist: float = 0.5,
    plot: bool = False,
) -> dict[str, float]:
    """Evaluate one prediction using the preserved FastVGGT ScanNet protocol."""

    validate_scene_inputs(scene)
    points, predicted_c2w, inference_seconds = _validate_prediction(scene, prediction)
    if isinstance(chamfer_max_dist, bool):
        raise FastVGGTEvaluationError("chamfer_max_dist must be finite and positive")
    try:
        max_dist = float(chamfer_max_dist)
    except (TypeError, ValueError) as error:
        raise FastVGGTEvaluationError("chamfer_max_dist must be finite and positive") from error
    if not np.isfinite(max_dist) or max_dist <= 0:
        raise FastVGGTEvaluationError("chamfer_max_dist must be finite and positive")

    gt_c2w = np.asarray(scene.poses_c2w, dtype=np.float64)
    first_gt_pose = gt_c2w[0].copy()
    normalized_gt_c2w = np.linalg.inv(first_gt_pose) @ gt_c2w
    predicted_w2c = np.linalg.inv(predicted_c2w)
    metrics = evaluate_scene_and_save(
        scene.scene_id,
        normalized_gt_c2w,
        first_gt_pose,
        list(scene.frame_ids),
        list(predicted_w2c),
        [points],
        Path(output_dir),
        Path(scene.gt_ply),
        max_dist,
        inference_seconds * 1000.0,
        bool(plot),
    )
    return _validated_metrics(metrics)


__all__ = [
    "FastVGGTEvaluationError",
    "PROTOCOL_ID",
    "evaluate_prediction",
    "validate_scene_inputs",
]
