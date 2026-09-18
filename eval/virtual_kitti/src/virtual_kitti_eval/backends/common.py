"""Validated wire contracts and audited native geometry conversion.

The native load/conversion helpers originate in the standalone ScanNet evaluator;
this package owns its copies and imports no sibling evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Protocol
import uuid
import zipfile

import numpy as np

from ..io import canonical_json
from ..provenance import JSONValue


def plain_json(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("JSON keys must be strings")
        return {k: plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    if value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("JSON value required")
    canonical_json(value)
    return value


def freeze_json(value):
    value = plain_json(value)
    if isinstance(value, dict):
        return MappingProxyType({k: freeze_json(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(v) for v in value)
    return value


def original_frame_ids(values):
    ids = tuple(values)
    if (not ids or any(not isinstance(v, str) or re.fullmatch(r"[0-9]{5}", v) is None for v in ids)
        or ids != tuple(sorted(set(ids)))):
        raise ValueError("BACKEND_FRAME_IDS: ordered unique original five-digit IDs required")
    return ids


def _real_array(value, label):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError("real integer or floating-point array required")
        return np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: real numeric array required") from exc


def validate_poses(poses, count):
    poses = _real_array(poses, "BACKEND_POSES")
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("BACKEND_POSES: finite (N,4,4) with matching count required")
    rotation = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4, rtol=0)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)):
        raise ValueError("BACKEND_POSES: proper rigid c2w required; scale/reflection forbidden")
    return poses


@dataclass(frozen=True)
class BackendPrediction:
    frame_ids: tuple[str, ...]
    poses_c2w: np.ndarray
    world_points: np.ndarray | None
    metadata: dict[str, JSONValue]

    def __post_init__(self):
        ids = original_frame_ids(self.frame_ids)
        object.__setattr__(self, "frame_ids", ids)
        object.__setattr__(self, "poses_c2w", validate_poses(self.poses_c2w, len(ids)))
        if self.world_points is not None:
            points = _real_array(self.world_points, "BACKEND_POINTS")
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError("BACKEND_POINTS: finite (M,3) world points required")
            object.__setattr__(self, "world_points", points)
        if (not isinstance(self.metadata, dict) or self.metadata.get("pose_convention") != "c2w"
            or self.metadata.get("pose_scale") != "rigid"):
            raise ValueError("BACKEND_CONVENTION: explicit rigid c2w metadata required")
        object.__setattr__(self, "metadata", plain_json(self.metadata))


def validate_prediction(prediction, frame_ids):
    if not isinstance(prediction, BackendPrediction):
        raise ValueError("BACKEND_OUTPUT: BackendPrediction required")
    prediction.__post_init__()
    if prediction.frame_ids != tuple(frame_ids):
        raise ValueError("BACKEND_FRAME_MISMATCH: output IDs/order differ from request")
    return prediction


@dataclass(frozen=True)
class BackendRequest:
    model_key: str
    model_config: Mapping[str, JSONValue]
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    output_dir: Path
    request_id: str
    provenance_id: str
    sequence: str
    device: str = "cuda:0"

    def __post_init__(self):
        from . import normalize_model_key
        object.__setattr__(self, "model_key", normalize_model_key(self.model_key))
        object.__setattr__(self, "frame_ids", original_frame_ids(self.frame_ids))
        paths = tuple(Path(p).absolute() for p in self.image_paths)
        if (len(paths) != len(self.frame_ids) or any(not p.is_file() for p in paths)
            or any(p.suffix.lower() not in (".png", ".jpg", ".jpeg") for p in paths)
            or any(p.stem != frame for p, frame in zip(paths, self.frame_ids))):
            raise ValueError("BACKEND_INPUT: matching decoded image paths required")
        object.__setattr__(self, "image_paths", paths)
        object.__setattr__(self, "output_dir", Path(self.output_dir).absolute())
        if not isinstance(self.model_config, Mapping):
            raise ValueError("BACKEND_CONFIG: mapping required")
        for name in ("interpreter", "project_root", "checkpoint"):
            if not isinstance(self.model_config.get(name), str) or not Path(self.model_config[name]).is_absolute():
                raise ValueError(f"BACKEND_CONFIG: absolute {name} required")
        object.__setattr__(self, "model_config", freeze_json(self.model_config))
        if (not isinstance(self.request_id, str) or not self.request_id
            or re.fullmatch("[a-f0-9]{64}", self.provenance_id) is None
            or re.fullmatch(r"Scene(01|02|06|18|20)/(Clone|Fog|Morning|Overcast|Rain|Sunset)", self.sequence) is None
            or re.fullmatch(r"cuda(?::[0-9]+)?", self.device) is None):
            raise ValueError("BACKEND_REQUEST: invalid identity, provenance, sequence or CUDA device")

    @classmethod
    def from_prepared(cls, prepared, *, model_key, model_config, output_dir, provenance_id,
                      request_id=None, device="cuda:0"):
        return cls(model_key, model_config, tuple(prepared.frame_ids), tuple(prepared.image_paths),
                   output_dir, request_id or uuid.uuid4().hex, provenance_id, prepared.sequence, device)

    def to_dict(self):
        return {"schema_version": 1, "model_key": self.model_key,
                "model_config": plain_json(self.model_config), "frame_ids": list(self.frame_ids),
                "image_paths": [str(p) for p in self.image_paths], "output_dir": str(self.output_dir),
                "request_id": self.request_id, "provenance_id": self.provenance_id,
                "sequence": self.sequence, "device": self.device}

    @classmethod
    def from_dict(cls, data):
        fields = {"schema_version", "model_key", "model_config", "frame_ids", "image_paths",
                  "output_dir", "request_id", "provenance_id", "sequence", "device"}
        if not isinstance(data, dict) or set(data) != fields or type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("BACKEND_REQUEST_SCHEMA")
        return cls(**{k: v for k, v in data.items() if k != "schema_version"})


class NativeBackend(Protocol):
    def load(self) -> None: ...
    def infer(self, frame_ids: tuple[str, ...], image_paths: tuple[Path, ...]) -> BackendPrediction: ...


def load_state(path):
    import torch
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("checkpoint must contain a nonempty tensor state dictionary")
    return state


def checked_load(model, state):
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(state) if expected[k].shape != state[k].shape)
    if missing or unexpected or mismatched:
        raise ValueError(f"incompatible checkpoint: missing={missing[:12]}, unexpected={unexpected[:12]}, shape={mismatched[:12]}")
    model.load_state_dict(state, strict=True)
    return {"loaded_keys": len(expected), "unused_keys": []}


def inspect_checkpoint(path):
    """Check on-disk container completeness without torch, pickle or CUDA allocation."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            first = stream.read(128)
            if first.startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError("Git LFS pointer; checkpoint bytes unavailable")
            if path.suffix == ".safetensors":
                length = int.from_bytes(first[:8], "little")
                if not 2 <= length <= min(size - 8, 100_000_000):
                    raise ValueError("invalid/truncated safetensors header")
                stream.seek(8)
                header = json.loads(stream.read(length))
                tensors = {k: v for k, v in header.items() if k != "__metadata__"}
                if not tensors:
                    raise ValueError("empty tensor inventory")
                widths = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                          "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
                          "F32": 4, "I64": 8, "U64": 8, "F64": 8}
                spans = []
                for tensor in tensors.values():
                    shape, offsets = tensor["shape"], tensor["data_offsets"]
                    if any(type(v) is not int or v < 0 for v in shape):
                        raise ValueError("invalid tensor shape")
                    start, end = offsets
                    if type(start) is not int or type(end) is not int or start < 0 or end - start != math.prod(shape) * widths[tensor["dtype"]]:
                        raise ValueError("invalid tensor byte span")
                    spans.append((start, end))
                last = 0
                for start, end in sorted(spans):
                    if start != last:
                        raise ValueError("noncontiguous tensor data")
                    last = end
                if 8 + length + last != size:
                    raise ValueError("truncated or trailing tensor data")
                return {"format": "safetensors", "size_bytes": size, "tensor_count": len(tensors)}
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or not any(n.endswith("/data.pkl") for n in names) or not any("/data/" in n for n in names):
                raise ValueError("not a complete PyTorch tensor archive")
            bad = archive.testzip()
            if bad:
                raise ValueError(f"corrupt archive member: {bad}")
            return {"format": "pytorch_zip", "size_bytes": size, "members": len(names), "crc_checked": True}
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, OverflowError) as exc:
        raise ValueError(f"CHECKPOINT_INVALID: {path}: {exc}") from exc


def w2c_to_c2w(extrinsics):
    extrinsics = _real_array(extrinsics, "BACKEND_POSES")
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("w2c must have shape [N,3,4] or [N,4,4]")
    matrices = np.broadcast_to(np.eye(4), (len(extrinsics), 4, 4)).copy()
    matrices[:, :extrinsics.shape[1]] = extrinsics
    return np.linalg.inv(validate_poses(matrices, len(matrices)))


def transform_projective(points, homography, epsilon=1e-9):
    points = np.asarray(points).reshape(-1, 3)
    h = np.asarray(homography)
    if h.shape != (4, 4) or not np.isfinite(h).all():
        raise ValueError("invalid graph homography")
    result = np.column_stack((points, np.ones(len(points)))) @ h.T
    valid = np.isfinite(result).all(axis=1) & (np.abs(result[:, 3]) > epsilon)
    output = np.full((len(points), 3), np.nan)
    output[valid] = result[valid, :3] / result[valid, 3, None]
    return output, valid


def extract_long_chunks(chunks, ranges, transforms, count, coefficient=.75):
    if len(chunks) != len(ranges) or len(transforms) != len(chunks) - 1:
        raise ValueError("native chunk transform count mismatch")
    owners = np.full(count, -1, dtype=int)
    for k, (start, end) in enumerate(ranges):
        if not 0 <= start < end <= count:
            raise ValueError("invalid native chunk range")
        owners[start:end] = k
    if np.any(owners < 0):
        raise ValueError("native Long omitted requested frames")
    cameras = np.empty((count, 4, 4))
    output, thresholds = [], []
    for k, (chunk, (start, end)) in enumerate(zip(chunks, ranges)):
        points = np.asarray(chunk["world_points"])
        conf = np.asarray(chunk["world_points_conf"])
        poses = validate_poses(chunk["extrinsic"], end - start)
        if len(points) != end - start or points.shape[:-1] != conf.shape:
            raise ValueError("native chunk frame/point/confidence mismatch")
        scale, rotation, translation = (1., np.eye(3), np.zeros(3)) if k == 0 else transforms[k - 1]
        scale, rotation, translation = float(scale), np.asarray(rotation), np.asarray(translation).reshape(3)
        rigid = np.eye(4)
        rigid[:3, :3], rigid[:3, 3] = rotation, translation
        validate_poses(rigid[None], 1)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid native Sim3")
        threshold = float(np.mean(conf) * coefficient)
        thresholds.append(threshold)
        if not np.isfinite(threshold):
            raise ValueError("nonfinite native confidence threshold")
        for j, index in enumerate(range(start, end)):
            if owners[index] != k:
                continue
            frame = points[j].reshape(-1, 3)
            mask = np.isfinite(conf[j].reshape(-1)) & (conf[j].reshape(-1) > threshold)
            output.append(scale * (frame[mask] @ rotation.T) + translation)
            cameras[index] = poses[j]
            cameras[index, :3, :3] = rotation @ poses[j, :3, :3]
            cameras[index, :3, 3] = scale * (rotation @ poses[j, :3, 3]) + translation
    return np.concatenate(output), cameras, {"frame_owner": owners.tolist(), "chunks": len(chunks),
        "confidence_rule": "conf > full_chunk_mean * coefficient", "conf_threshold_coef": coefficient,
        "confidence_thresholds": thresholds}


def extract_slam_submaps(submaps, graph, path_ids, ids):
    owned, output = {}, []
    invalid, regular = 0, 0
    for submap in submaps:
        if submap.get_lc_status():
            continue
        regular += 1
        poses = submap.get_all_poses_world(graph)
        if len(poses) != len(submap.img_names) or len(submap.pointclouds) != len(poses):
            raise ValueError("SLAM submap frame count mismatch")
        for j, name in enumerate(submap.img_names):
            if str(name) not in path_ids:
                raise ValueError("SLAM emitted unrequested image")
            frame_id = path_ids[str(name)]
            if frame_id in owned:
                continue
            points, valid = transform_projective(submap.pointclouds[j], graph.get_homography(submap.get_id() + j))
            conf = np.asarray(submap.conf_masks[j]).reshape(-1)
            if len(conf) != len(points):
                raise ValueError("SLAM confidence shape mismatch")
            mask = np.isfinite(conf) & (conf > submap.conf_threshold)
            invalid += int((mask & ~valid).sum())
            output.append(points[mask & valid])
            owned[frame_id] = (poses[j], submap.get_id())
    if set(owned) != set(ids):
        raise ValueError("SLAM frame coverage differs from request")
    return np.concatenate(output), np.stack([owned[i][0] for i in ids]), {
        "frame_owner": [owned[i][1] for i in ids], "submaps": regular,
        "projective_invalid_removed": invalid, "confidence_rule": "native submap depth_conf percentile + 1e-6"}


def stage_images(ids, paths, work_dir):
    directory = Path(work_dir) / "input_frames"
    directory.mkdir(exist_ok=False)
    staged = []
    for index, path in enumerate(paths):
        suffix = Path(path).suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png"):
            raise ValueError("native backend requires JPEG/PNG input")
        destination = directory / f"{index:08d}{'.jpg' if suffix == '.jpeg' else suffix}"
        destination.symlink_to(path)
        staged.append(str(destination))
    (directory / "frame_ids.json").write_text(json.dumps({
        "frame_ids": list(ids), "source_paths": [str(p) for p in paths]}, indent=2))
    return directory, staged
