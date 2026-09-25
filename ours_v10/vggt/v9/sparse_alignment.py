"""Opt-in sparse point plus camera Sim(3) on frozen v8 window predictions.

The v8 optimizer, camera residuals, Huber loss, transform direction and
front-window ownership are reused. The only changed fitting input is a
deterministic 16x16 spatial selection of same-frame, same-pixel point pairs.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.ours_v3.geometry import Sim3, append_unique, transform_predictions
from experiments.ours_v3.stitch import json_write
from vggt.v5.alignment import LongAlignmentConfig, _nondegenerate, load_long, validate_prediction
from vggt.v8.joint_alignment import (
    AlignmentStitcher as V8AlignmentStitcher,
    JointAlignmentConfig,
    _cameras,
    _components,
    _optimize,
    align_overlap_joint as full_align_overlap_joint,
)


class SparseAlignmentRejected(ValueError):
    """A declared sparse-data or geometry gate failed; full alignment may run."""


@dataclass(frozen=True)
class SparseAlignmentConfig:
    joint: JointAlignmentConfig = field(default_factory=lambda: JointAlignmentConfig(
        mode="point_camera_joint"))
    initializer: LongAlignmentConfig = field(default_factory=LongAlignmentConfig)
    grid_size: int = 16
    min_cells_per_frame: int = 32
    require_all_quadrants: bool = True
    min_scale: float = .1
    max_scale: float = 10.
    max_objective_increase: float = 1e-8

    def __post_init__(self):
        if self.joint.mode != "point_camera_joint":
            raise ValueError("sparse mode requires v8 point_camera_joint objective")
        if self.grid_size != 16:
            raise ValueError("v9 requires a fixed 16x16 grid")
        if not 3 <= self.min_cells_per_frame <= 256:
            raise ValueError("invalid minimum spatial cell count")
        if (not 0 < self.min_scale < self.max_scale or
                self.max_objective_increase < 0):
            raise ValueError("invalid sparse transform gates")


@dataclass
class SelectedCorrespondences:
    frame_ids: list
    indices: list
    source: np.ndarray
    target: np.ndarray
    weights: np.ndarray
    confidence_a: np.ndarray
    confidence_b: np.ndarray
    confidence_threshold: float
    candidate_per_frame: list
    selected_per_frame: list
    quadrants_per_frame: list
    index_sha256: str
    center_a: np.ndarray
    center_b: np.ndarray
    rotation_a: np.ndarray
    rotation_b: np.ndarray
    geometry_scale: float

    def optimizer_data(self):
        return dict(common_frame_ids=self.frame_ids, source=self.source,
                    target=self.target, weights=self.weights,
                    center_a=self.center_a, center_b=self.center_b,
                    rotation_a=self.rotation_a, rotation_b=self.rotation_b,
                    geometry_scale=self.geometry_scale,
                    confidence_threshold=self.confidence_threshold)

    def records(self):
        return [dict(frame_id=str(frame), row=int(row), col=int(col),
                     confidence_a=float(ca), confidence_b=float(cb),
                     joint_confidence=float(np.sqrt(ca * cb)))
                for (frame, row, col), ca, cb in zip(
                    self.indices, self.confidence_a, self.confidence_b)]


def _prediction_arrays(prediction):
    ids = list(prediction["frame_ids"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate frame IDs")
    points = np.asarray(prediction["world_points"])
    confidence = np.asarray(prediction["world_points_conf"])
    if (points.ndim != 4 or points.shape[-1] != 3 or
            points.shape[0] != len(ids) or confidence.shape != points.shape[:-1]):
        raise ValueError("point/confidence shape mismatch")
    if not np.isfinite(confidence).all() or (confidence < 0).any():
        raise ValueError("nonfinite or negative confidence")
    return ids, points, confidence


def select_sparse_correspondences(a, b, config):
    """One max-joint-confidence valid pair per grid cell and overlap frame.

    Ties use the first row-major pixel. A grid cell is determined by original
    pixel coordinates, never by a packed or re-numbered point index.
    """
    ai, ap, ac = _prediction_arrays(a)
    bi, bp, bc = _prediction_arrays(b)
    lookup_a = {frame: index for index, frame in enumerate(ai)}
    lookup_b = {frame: index for index, frame in enumerate(bi)}
    common = [frame for frame in ai if frame in lookup_b]
    if not common:
        raise SparseAlignmentRejected("no common frame IDs")
    ia = [lookup_a[frame] for frame in common]
    ib = [lookup_b[frame] for frame in common]
    if ap.shape[1:3] != bp.shape[1:3]:
        raise SparseAlignmentRejected("overlap pixel grids differ")
    threshold = float(.1 * min(np.median(ac[ia]), np.median(bc[ib])))
    height, width = ap.shape[1:3]
    if height < config.grid_size or width < config.grid_size:
        raise SparseAlignmentRejected("image is smaller than 16x16 selection grid")
    chosen = []
    source, target, weights, confidence_a, confidence_b = [], [], [], [], []
    candidates_per_frame, selected_per_frame, quadrants_per_frame = [], [], []
    for frame, index_a, index_b in zip(common, ia, ib):
        pts_a, pts_b = ap[index_a], bp[index_b]
        conf_a, conf_b = ac[index_a], bc[index_b]
        valid = ((conf_a > threshold) & (conf_b > threshold) &
                 np.isfinite(pts_a).all(axis=-1) & np.isfinite(pts_b).all(axis=-1))
        candidates_per_frame.append(int(valid.sum()))
        frame_indices = []
        quadrants = set()
        for grid_row in range(config.grid_size):
            row_start = grid_row * height // config.grid_size
            row_end = (grid_row + 1) * height // config.grid_size
            for grid_col in range(config.grid_size):
                col_start = grid_col * width // config.grid_size
                col_end = (grid_col + 1) * width // config.grid_size
                cell_valid = valid[row_start:row_end, col_start:col_end]
                if not cell_valid.any():
                    continue
                product = (conf_a[row_start:row_end, col_start:col_end].astype(np.float64) *
                           conf_b[row_start:row_end, col_start:col_end].astype(np.float64))
                product = np.where(cell_valid, product, -np.inf)
                dr, dc = np.unravel_index(int(np.argmax(product)), product.shape)
                row, col = row_start + int(dr), col_start + int(dc)
                frame_indices.append((frame, row, col))
                quadrants.add((int(grid_row >= 8), int(grid_col >= 8)))
                source.append(np.asarray(pts_b[row, col], dtype=np.float64))
                target.append(np.asarray(pts_a[row, col], dtype=np.float64))
                left, right = float(conf_a[row, col]), float(conf_b[row, col])
                confidence_a.append(left)
                confidence_b.append(right)
                weights.append(float(np.sqrt(left * right)))
        selected_per_frame.append(len(frame_indices))
        quadrants_per_frame.append(len(quadrants))
        chosen.extend(frame_indices)
        if len(frame_indices) < config.min_cells_per_frame or (
                config.require_all_quadrants and len(quadrants) != 4):
            raise SparseAlignmentRejected(
                f"insufficient spatial coverage for frame {frame}: "
                f"cells={len(frame_indices)}, quadrants={len(quadrants)}")
    if len(source) < 3:
        raise SparseAlignmentRejected("fewer than three sparse point pairs")
    source = np.stack(source); target = np.stack(target)
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise SparseAlignmentRejected("invalid selected joint confidence")
    _nondegenerate(source, config.initializer.degeneracy_ratio)
    _nondegenerate(target, config.initializer.degeneracy_ratio)
    center = np.median(target, axis=0)
    geometry_scale = float(np.median(np.linalg.norm(target - center, axis=1)))
    if not np.isfinite(geometry_scale) or geometry_scale <= 1e-8:
        raise SparseAlignmentRejected("degenerate sparse geometry scale")
    center_a, rotation_a = _cameras(a, ia)
    center_b, rotation_b = _cameras(b, ib)
    canonical = [[str(frame), int(row), int(col)] for frame, row, col in chosen]
    digest = hashlib.sha256(json.dumps(canonical, separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8")).hexdigest()
    return SelectedCorrespondences(
        frame_ids=common, indices=chosen, source=source, target=target,
        weights=weights, confidence_a=np.asarray(confidence_a),
        confidence_b=np.asarray(confidence_b), confidence_threshold=threshold,
        candidate_per_frame=candidates_per_frame,
        selected_per_frame=selected_per_frame,
        quadrants_per_frame=quadrants_per_frame,
        index_sha256=digest, center_a=center_a, center_b=center_b,
        rotation_a=rotation_a, rotation_b=rotation_b,
        geometry_scale=geometry_scale)


def _sparse_initializer(selected, config):
    """Run Long's unchanged weighted IRLS core on selected pairs only."""
    long_config = config.initializer
    function = (load_long().robust_weighted_estimate_sim3_numba
                if long_config.align_method == "numba"
                else load_long().robust_weighted_estimate_sim3)
    scale, rotation, translation = function(
        selected.source, selected.target, selected.weights,
        delta=long_config.delta, max_iters=long_config.max_iters,
        tol=long_config.tol, using_sim3=True)
    if not np.isfinite(scale) or scale <= 0:
        raise SparseAlignmentRejected("nonfinite or nonpositive sparse initial scale")
    return Sim3(float(scale), np.asarray(rotation), np.asarray(translation))


def align_overlap_sparse(a, b, config=None):
    """Return B_local->A_local Sim(3), with explicit full-joint fallback."""
    config = config or SparseAlignmentConfig()
    started = time.perf_counter()
    timings = dict(selection=0., initialization=0., optimization=0.,
                   validation=0., fallback_full_seconds=0.)
    selected = None
    try:
        phase = time.perf_counter()
        selected = select_sparse_correspondences(a, b, config)
        data = selected.optimizer_data()
        timings["selection"] = time.perf_counter() - phase
        phase = time.perf_counter()
        initial = _sparse_initializer(selected, config)
        timings["initialization"] = time.perf_counter() - phase
        phase = time.perf_counter()
        before = _components(initial, data, config.joint, config.joint.chunk_size)
        transform, optimization = _optimize(initial, data, config.joint)
        after = _components(transform, data, config.joint, config.joint.chunk_size)
        timings["optimization"] = time.perf_counter() - phase
        phase = time.perf_counter()
        values = [*before.values(), *after.values()]
        if not np.isfinite(values).all():
            raise SparseAlignmentRejected("nonfinite sparse objective/residual")
        if not config.min_scale < transform.scale < config.max_scale:
            raise SparseAlignmentRejected("sparse scale outside predeclared range")
        if after["total"] > before["total"] + config.max_objective_increase:
            raise SparseAlignmentRejected("sparse objective increased")
        timings["validation"] = time.perf_counter() - phase
        timings["total_alignment"] = time.perf_counter() - started
        return transform, dict(
            alignment_mode="sparse_point_camera_joint", fallback=False,
            direction="B_local -> A_local", common_frame_ids=selected.frame_ids,
            pairs=len(selected.source), selected_pairs=len(selected.source),
            candidate_per_frame=selected.candidate_per_frame,
            selected_per_frame=selected.selected_per_frame,
            quadrants_per_frame=selected.quadrants_per_frame,
            confidence_threshold=selected.confidence_threshold,
            index_sha256=selected.index_sha256,
            selected_pixels=selected.records(),
            geometry_scale=selected.geometry_scale,
            initial_transform=initial.record(), final_transform=transform.record(),
            initial_loss=before, final_loss=after, optimization=optimization,
            timing_seconds=timings, config=asdict(config))
    except (SparseAlignmentRejected, ValueError, RuntimeError,
            np.linalg.LinAlgError, FloatingPointError) as error:
        reason = f"{type(error).__name__}: {error}"
        phase = time.perf_counter()
        full_transform, full_stats = full_align_overlap_joint(a, b, config.joint)
        timings["fallback_full_seconds"] = time.perf_counter() - phase
        timings["total_alignment"] = time.perf_counter() - started
        stats = dict(full_stats)
        stats.update(
            alignment_mode="sparse_point_camera_joint",
            fallback=True, fallback_reason=reason,
            direction="B_local -> A_local",
            selected_pairs=0 if selected is None else len(selected.source),
            candidate_per_frame=[] if selected is None else selected.candidate_per_frame,
            selected_per_frame=[] if selected is None else selected.selected_per_frame,
            quadrants_per_frame=[] if selected is None else selected.quadrants_per_frame,
            index_sha256=None if selected is None else selected.index_sha256,
            selected_pixels=[] if selected is None else selected.records(),
            timing_seconds=timings, sparse_config=asdict(config))
        return full_transform, stats


class V9AlignmentStitcher(V8AlignmentStitcher):
    """Use the exact v8 stitcher for full mode; override only sparse edge fit."""
    def __init__(self, directory, config=None):
        self.sparse = isinstance(config, SparseAlignmentConfig)
        super().__init__(directory, config or JointAlignmentConfig(mode="point_camera_joint"))

    def add(self, prediction, window_id):
        if not self.sparse:
            return super().add(prediction, window_id)
        edge = self.directory / f"edge_{window_id-1:04d}_{window_id:04d}.json"
        try:
            validate_prediction(prediction)
            if window_id != len(list(self.directory.glob("window_*_transform.json"))):
                raise ValueError("nonsequential window ID")
            if self.previous is not None:
                adjacent, stats = align_overlap_sparse(self.previous, prediction, self.config)
                self.global_transform = self.global_transform.compose(adjacent)
                record = dict(status="success", direction="B_local -> A_local",
                              composition="S_B_global = S_A_global compose S_B_to_A",
                              adjacent=adjacent.record(),
                              global_transform=self.global_transform.record())
                record.update(stats)
                json_write(edge, record)
            pose, depth = transform_predictions(
                prediction["c2w"], prediction["depth"], self.global_transform)
            if not np.isfinite(pose).all() or not np.isfinite(depth).all():
                raise ValueError("nonfinite transformed camera/depth")
            points = self.global_transform.apply(prediction["world_points"])
            fresh = append_unique(self.seen, list(prediction["frame_ids"]))
            for index in fresh:
                self.frames.append(prediction["frame_ids"][index])
                self.poses.append(pose[index])
                self.intrinsics.append(prediction["intrinsics"][index])
                self.sources.append(window_id)
            json_write(self.directory / f"window_{window_id:04d}_transform.json",
                       dict(direction="window_local -> global",
                            **self.global_transform.record(),
                            appended_frame_ids=[prediction["frame_ids"][index]
                                                for index in fresh]))
            self.previous = prediction
            transformed = {key: value for key, value in prediction.items()
                           if key != "pose_encoding"}
            transformed.update(c2w=pose, depth=depth, world_points=points)
            return fresh, transformed
        except Exception as error:
            json_write(edge, dict(status="failed", reason=str(error),
                                  traceback=traceback.format_exc()))
            raise
