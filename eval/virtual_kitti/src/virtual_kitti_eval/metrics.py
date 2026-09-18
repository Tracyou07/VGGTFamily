"""Virtual KITTI 1.3.1 camera-center ATE in metres after a proper, positive-scale Sim(3)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

PROTOCOL_ID = "virtual-kitti-1.3.1-ate-sim3-v1"


@dataclass(frozen=True)
class MatchedTrajectory:
    frame_ids: tuple[str, ...]
    pred_c2w: np.ndarray
    gt_c2w: np.ndarray


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(frozen=True)
class AteMetrics:
    protocol_id: str
    matched_frames: int
    rmse_m: float
    alignment: Sim3


def validate_frame_ids(ids) -> tuple[str, ...]:
    ids = tuple(ids)
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("FRAME_ID_INVALID: IDs must be unique nonempty strings")
    return ids


def _real_array(value):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError("REAL_NUMERIC_ARRAY_REQUIRED")
    try:
        return np.asarray(array, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("INVALID_NUMERIC_ARRAY") from exc


def _poses(value, count):
    value = _real_array(value)
    if value.shape != (count, 4, 4):
        raise ValueError("POSE_SHAPE: expected (N,4,4) c2w")
    if not np.isfinite(value).all():
        raise ValueError("NONFINITE_POSE")
    rotation = value[:, :3, :3]
    if (not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
        raise ValueError("POSE_INVALID: c2w must be rigid and orientation preserving")
    return value


def match_poses_by_frame_id(pred_ids, pred_c2w, gt_ids, gt_c2w) -> MatchedTrajectory:
    pred_ids, gt_ids = validate_frame_ids(pred_ids), validate_frame_ids(gt_ids)
    if set(pred_ids) != set(gt_ids):
        raise ValueError("FRAME_ID_MISMATCH: exact ID sets required")
    pred, gt = _poses(pred_c2w, len(pred_ids)), _poses(gt_c2w, len(gt_ids))
    index = {frame: i for i, frame in enumerate(pred_ids)}
    return MatchedTrajectory(gt_ids, pred[[index[frame] for frame in gt_ids]].copy(), gt.copy())


def umeyama_sim3(pred_xyz, gt_xyz) -> Sim3:
    pred, gt = _real_array(pred_xyz), _real_array(gt_xyz)
    if pred.ndim != 2 or pred.shape[1:] != (3,) or gt.shape != pred.shape:
        raise ValueError("TRAJECTORY_SHAPE: expected matching (N,3)")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("NONFINITE_TRAJECTORY")
    if len(pred) < 3:
        raise ValueError("DEGENERATE_TRAJECTORY: at least three non-collinear points required")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            pred_mean, gt_mean = pred.mean(axis=0), gt.mean(axis=0)
            x, y = pred - pred_mean, gt - gt_mean
            if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
                raise ValueError("DEGENERATE_TRAJECTORY: non-collinear points required")
            covariance = y.T @ x / len(x)
            u, singular, vt = np.linalg.svd(covariance)
            rank = np.linalg.matrix_rank(covariance)
            if rank < 2:
                raise ValueError("DEGENERATE_TRAJECTORY: rank-deficient alignment")
            sign = np.linalg.det(u @ vt)
            if rank == 3 and sign < 0:
                raise ValueError("REFLECTION: unconstrained fit reverses orientation")
            correction = np.diag([1., 1., 1. if sign >= 0 else -1.])
            rotation = u @ correction @ vt
            scale = float(np.sum(singular * np.diag(correction)) / np.mean(np.sum(x * x, axis=1)))
            translation = gt_mean - scale * rotation @ pred_mean
            if not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
                raise ValueError("DEGENERATE_SIM3")
            return Sim3(scale, rotation, translation)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError("NONFINITE_OR_UNSTABLE_ALIGNMENT") from exc


def apply_sim3(sim3: Sim3, xyz) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = sim3.scale * np.asarray(xyz, dtype=float) @ sim3.rotation.T + sim3.translation
            if not np.isfinite(result).all():
                raise ValueError("NONFINITE_ALIGNED_TRAJECTORY")
            return result
        except FloatingPointError as exc:
            raise ValueError("NONFINITE_ALIGNED_TRAJECTORY") from exc


def ate_rmse_m(pred_ids, pred_c2w, gt_ids, gt_c2w) -> AteMetrics:
    matched = match_poses_by_frame_id(pred_ids, pred_c2w, gt_ids, gt_c2w)
    pred, gt = matched.pred_c2w[:, :3, 3], matched.gt_c2w[:, :3, 3]
    sim3 = umeyama_sim3(pred, gt)
    try:
        with np.errstate(over="raise", invalid="raise"):
            residual = apply_sim3(sim3, pred) - gt
            rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    except FloatingPointError as exc:
        raise ValueError("NONFINITE_RMSE") from exc
    if not np.isfinite(rmse):
        raise ValueError("NONFINITE_RMSE")
    return AteMetrics(PROTOCOL_ID, len(matched.frame_ids), rmse, sim3)
