"""Opt-in normalized point and point-camera Sim(3) alignment for v8."""
from dataclasses import asdict, dataclass
from pathlib import Path
import traceback

import numpy as np
import torch
from scipy.optimize import minimize

from experiments.ours_v3.geometry import Sim3, append_unique, transform_predictions
from experiments.ours_v3.stitch import json_write
from vggt.v5.alignment import (LongAlignmentConfig, _nondegenerate,
                               align_overlap as legacy_align_overlap,
                               validate_prediction)

ALIGNMENT_MODES = ("point_legacy", "point_normalized_control", "point_camera_joint")


@dataclass(frozen=True)
class JointAlignmentConfig:
    mode: str = "point_legacy"
    huber_delta: float = .1
    lambda_center: float = 1.
    lambda_rotation: float = 1.
    chunk_size: int = 250_000
    max_iterations: int = 30
    tolerance: float = 1e-8

    def __post_init__(self):
        if self.mode not in ALIGNMENT_MODES:
            raise ValueError("unknown alignment mode")
        if (self.huber_delta <= 0 or self.lambda_center < 0 or
                self.lambda_rotation < 0 or self.chunk_size < 1 or
                self.max_iterations < 1 or self.tolerance <= 0):
            raise ValueError("invalid joint alignment configuration")


def _cameras(prediction, indices):
    poses = np.asarray(prediction["c2w"], dtype=np.float64)[indices]
    if poses.shape != (len(indices), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("nonfinite or invalid c2w cameras")
    rotations = poses[:, :3, :3]
    if (not np.allclose(np.einsum("fji,fjk->fik", rotations, rotations), np.eye(3), atol=5e-4) or
            not np.allclose(np.linalg.det(rotations), 1., atol=5e-4)):
        raise ValueError("invalid c2w rotations")
    return poses[:, :3, 3], rotations


def _prepare(a, b):
    ai, ap, ac = validate_prediction(a); bi, bp, bc = validate_prediction(b)
    lookup = {frame: i for i, frame in enumerate(bi)}
    common = [frame for frame in ai if frame in lookup]
    if not common:
        raise ValueError("no common frame IDs")
    ia = [ai.index(frame) for frame in common]; ib = [lookup[frame] for frame in common]
    ap, ac, bp, bc = ap[ia], ac[ia], bp[ib], bc[ib]
    if ap.shape != bp.shape:
        raise ValueError("overlap pixel grids differ")
    threshold = float(.1 * min(np.median(ac), np.median(bc)))
    mask = (ac > threshold) & (bc > threshold)
    target = np.asarray(ap[mask], dtype=np.float64)
    source = np.asarray(bp[mask], dtype=np.float64)
    weights = np.sqrt(np.asarray(ac[mask], dtype=np.float64) *
                      np.asarray(bc[mask], dtype=np.float64))
    if len(source) < 3 or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("insufficient valid overlap correspondences")
    _nondegenerate(source, 1e-6); _nondegenerate(target, 1e-6)
    center = np.median(target, axis=0)
    length = float(np.median(np.linalg.norm(target - center, axis=1)))
    if not np.isfinite(length) or length <= 1e-8:
        raise ValueError("degenerate frozen geometry scale")
    ca, ra = _cameras(a, ia); cb, rb = _cameras(b, ib)
    return dict(common_frame_ids=common, source=source, target=target,
                weights=weights, center_a=ca, center_b=cb,
                rotation_a=ra, rotation_b=rb, geometry_scale=length,
                confidence_threshold=threshold)


def _huber(values, delta):
    return np.where(values <= delta, .5 * values**2,
                    delta * (values - .5 * delta))


def _components(transform, data, config, chunk_size):
    point_sum = 0.; weight_sum = float(data["weights"].sum())
    raw_sum = 0.; raw_square = 0.; count = len(data["source"])
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        residual = np.linalg.norm(transform.apply(data["source"][start:stop]) -
                                  data["target"][start:stop], axis=1)
        normalized = residual / data["geometry_scale"]
        weights = data["weights"][start:stop]
        point_sum += float(np.sum(weights * _huber(normalized, config.huber_delta)))
        raw_sum += float(residual.sum()); raw_square += float(np.square(residual).sum())
    centers = np.linalg.norm(transform.apply(data["center_b"]) - data["center_a"], axis=1)
    relative = np.einsum("fji,jk,fkl->fil", data["rotation_a"],
                         transform.rotation, data["rotation_b"])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.) / 2., -1., 1.)
    angles = np.arccos(cosine)
    center_loss = float(_huber(centers / data["geometry_scale"], config.huber_delta).mean())
    rotation_loss = float(_huber(angles, config.huber_delta).mean())
    center_weight = config.lambda_center if config.mode == "point_camera_joint" else 0.
    rotation_weight = config.lambda_rotation if config.mode == "point_camera_joint" else 0.
    point_loss = point_sum / weight_sum
    return dict(point=point_loss, center=center_loss, rotation=rotation_loss,
                total=point_loss + center_weight * center_loss + rotation_weight * rotation_loss,
                point_residual_mean=raw_sum/count,
                point_residual_rmse=float(np.sqrt(raw_square/count)),
                center_residual_mean=float(centers.mean()),
                rotation_residual_mean_rad=float(angles.mean()))


def loss_components(transform, a, b, config, chunk_size=None):
    data = _prepare(a, b)
    return _components(transform, data, config, chunk_size or config.chunk_size)


def _skew(vector):
    zero = vector.new_zeros(())
    x, y, z = vector.unbind()
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)


def _so3_exp(vector):
    theta = torch.linalg.vector_norm(vector)
    matrix = _skew(vector)
    a = torch.sinc(theta / torch.pi)
    b = .5 * torch.sinc(theta / (2 * torch.pi)) ** 2
    return torch.eye(3, dtype=vector.dtype) + a * matrix + b * (matrix @ matrix)


def _optimize(initial, data, config):
    source = torch.from_numpy(data["source"]); target = torch.from_numpy(data["target"])
    weights = torch.from_numpy(data["weights"]); weight_sum = weights.sum()
    ca = torch.from_numpy(data["center_a"]); cb = torch.from_numpy(data["center_b"])
    ra = torch.from_numpy(data["rotation_a"]); rb = torch.from_numpy(data["rotation_b"])
    r0 = torch.from_numpy(np.asarray(initial.rotation, dtype=np.float64))
    t0 = torch.from_numpy(np.asarray(initial.translation, dtype=np.float64))
    length = data["geometry_scale"]
    center_weight = config.lambda_center if config.mode == "point_camera_joint" else 0.
    rotation_weight = config.lambda_rotation if config.mode == "point_camera_joint" else 0.

    def transform(parameters):
        scale = float(initial.scale) * torch.exp(parameters[0])
        rotation = _so3_exp(parameters[1:4]) @ r0
        translation = t0 + parameters[4:7]
        return scale, rotation, translation

    def huber(values):
        delta = config.huber_delta
        return torch.where(values <= delta, .5 * values.square(),
                           delta * (values - .5 * delta))

    evaluations = 0
    def objective(array):
        nonlocal evaluations
        evaluations += 1
        parameters = torch.tensor(array, dtype=torch.float64, requires_grad=True)
        total_value = 0.
        for start in range(0, len(source), config.chunk_size):
            stop = min(start + config.chunk_size, len(source))
            scale, rotation, translation = transform(parameters)
            predicted = scale * (source[start:stop] @ rotation.T) + translation
            residual = torch.linalg.vector_norm(predicted - target[start:stop], dim=1) / length
            loss = (weights[start:stop] * huber(residual)).sum() / weight_sum
            loss.backward(); total_value += float(loss.detach())
        if center_weight or rotation_weight:
            scale, rotation, translation = transform(parameters)
            center_residual = torch.linalg.vector_norm(
                scale * (cb @ rotation.T) + translation - ca, dim=1) / length
            relative = ra.transpose(1, 2) @ rotation @ rb
            cosine = ((relative.diagonal(dim1=1, dim2=2).sum(1) - 1.) / 2.).clamp(-1+1e-12, 1-1e-12)
            angle = torch.acos(cosine)
            camera_loss = center_weight * huber(center_residual).mean() + \
                          rotation_weight * huber(angle).mean()
            camera_loss.backward(); total_value += float(camera_loss.detach())
        gradient = parameters.grad.detach().numpy().copy()
        return total_value, gradient

    result = minimize(objective, np.zeros(7), method="L-BFGS-B", jac=True,
                      options=dict(maxiter=config.max_iterations, ftol=1e-12,
                                   gtol=config.tolerance, maxls=30))
    scale, rotation, translation = transform(torch.from_numpy(result.x))
    candidate = Sim3(float(scale), rotation.numpy(), translation.numpy())
    if (not result.success or not np.isfinite(candidate.scale) or
            not 1e-6 < candidate.scale < 1e6 or
            not np.isfinite(candidate.rotation).all() or
            not np.isfinite(candidate.translation).all() or
            not np.allclose(candidate.rotation.T @ candidate.rotation, np.eye(3), atol=5e-6) or
            not np.isclose(np.linalg.det(candidate.rotation), 1., atol=5e-6)):
        raise RuntimeError(f"joint alignment optimization failed: {result.message}")
    return candidate, dict(success=bool(result.success), status=int(result.status),
                           message=str(result.message), iterations=int(result.nit),
                           function_evaluations=int(result.nfev), objective_evaluations=evaluations,
                           gradient_norm=float(np.linalg.norm(result.jac)))


def align_overlap_joint(a, b, config):
    if config.mode == "point_legacy":
        transform, stats = legacy_align_overlap(a, b, LongAlignmentConfig())
        stats.update(alignment_mode=config.mode, objective="legacy Long point-map IRLS")
        return transform, stats
    data = _prepare(a, b)
    initial, legacy_stats = legacy_align_overlap(a, b, LongAlignmentConfig())
    before = _components(initial, data, config, config.chunk_size)
    transform, optimization = _optimize(initial, data, config)
    after = _components(transform, data, config, config.chunk_size)
    return transform, dict(alignment_mode=config.mode,
        direction="B_local -> A_local", common_frame_ids=data["common_frame_ids"],
        pairs=len(data["source"]), confidence_threshold=data["confidence_threshold"],
        geometry_scale=data["geometry_scale"], geometry_scale_unit="point-head coordinate units",
        point_residual_unit="point-head coordinate units; objective divides by geometry_scale",
        weights=dict(point=1., center=config.lambda_center if config.mode=="point_camera_joint" else 0.,
                     rotation=config.lambda_rotation if config.mode=="point_camera_joint" else 0.),
        huber_delta=config.huber_delta, initial_transform=initial.record(),
        final_transform=transform.record(), initial_loss=before, final_loss=after,
        legacy_initialization=legacy_stats, optimization=optimization,
        config=asdict(config))


class AlignmentStitcher:
    def __init__(self, directory, config=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.config=config or JointAlignmentConfig()
        self.previous=None;self.global_transform=Sim3(1.,np.eye(3),np.zeros(3))
        self.seen=set();self.frames=[];self.poses=[];self.intrinsics=[];self.sources=[]

    def add(self, prediction, window_id):
        edge=self.directory/f"edge_{window_id-1:04d}_{window_id:04d}.json"
        try:
            validate_prediction(prediction)
            if window_id != len(list(self.directory.glob("window_*_transform.json"))):
                raise ValueError("nonsequential window ID")
            if self.previous is not None:
                adjacent,stats=align_overlap_joint(self.previous,prediction,self.config)
                self.global_transform=self.global_transform.compose(adjacent)
                record=dict(status="success",direction="B_local -> A_local",
                    composition="S_B_global = S_A_global compose S_B_to_A",
                    adjacent=adjacent.record(),global_transform=self.global_transform.record())
                record.update(stats)
                json_write(edge,record)
            pose,depth=transform_predictions(prediction["c2w"],prediction["depth"],self.global_transform)
            if not np.isfinite(pose).all() or not np.isfinite(depth).all():
                raise ValueError("nonfinite transformed camera/depth")
            points=self.global_transform.apply(prediction["world_points"])
            fresh=append_unique(self.seen,list(prediction["frame_ids"]))
            for i in fresh:
                self.frames.append(prediction["frame_ids"][i]);self.poses.append(pose[i])
                self.intrinsics.append(prediction["intrinsics"][i]);self.sources.append(window_id)
            json_write(self.directory/f"window_{window_id:04d}_transform.json",
                dict(direction="window_local -> global",**self.global_transform.record(),
                     appended_frame_ids=[prediction["frame_ids"][i] for i in fresh]))
            self.previous=prediction
            transformed={k:v for k,v in prediction.items() if k!="pose_encoding"}
            transformed.update(c2w=pose,depth=depth,world_points=points)
            return fresh,transformed
        except Exception as error:
            json_write(edge,dict(status="failed",reason=str(error),traceback=traceback.format_exc()))
            raise

    def finish(self, expected_ids):
        if self.frames != list(expected_ids):
            raise ValueError("final frame mapping is not exact")
        return dict(frame_ids=np.asarray(self.frames),c2w=np.asarray(self.poses),
                    intrinsics=np.asarray(self.intrinsics),source_window=np.asarray(self.sources))
