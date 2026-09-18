"""Strict configuration for the documented monocular Virtual KITTI 1.3.1 release."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import re
from .io import read_json

PROTOCOL_ID = "virtual-kitti-1.3.1-monocular-c2w-v1"
CONDITIONS = ("Clone", "Fog", "Morning", "Overcast", "Rain", "Sunset")
CANONICAL_SEQUENCES = tuple(f"Scene{s}/{c}" for s in ("01", "02") for c in CONDITIONS)
ALLOWED_SEQUENCES = tuple(f"Scene{s}/{c}" for s in ("01", "02", "06", "18", "20") for c in CONDITIONS)

class DatasetValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(f"{code}: {message}")

@dataclass(frozen=True)
class VirtualKittiConfig:
    schema_version: int
    raw_root: Path
    archive_root: Path
    prepared_root: Path
    sequences_file: Path
    sequence_ids: tuple[str, ...]
    dataset_version: str = "1.3.1"
    camera: str = "monocular"
    models: Mapping = field(default_factory=lambda: MappingProxyType({}))

def validate_sequence_ids(ids):
    ids = tuple(ids)
    if (not ids or any(s not in ALLOWED_SEQUENCES for s in ids)
        or len(set(ids)) != len(ids)):
        raise DatasetValidationError("INVALID_SEQUENCE", "Use unique canonical SceneXX/Condition IDs")
    return ids

def read_sequence_ids(path: Path):
    try:
        return validate_sequence_ids(line.strip() for line in path.read_text().splitlines() if line.strip())
    except OSError as exc:
        raise DatasetValidationError("INVALID_SEQUENCE", str(exc)) from exc

def load_config(path: Path) -> VirtualKittiConfig:
    path = Path(path).resolve()
    try:
        values = read_json(path)
        required = {"schema_version", "dataset_version", "camera", "raw_root", "archive_root",
                    "prepared_root", "sequences_file"}
        if not required.issubset(values) or set(values) - required - {"models", "gpu_min_free_mib", "gpu_require_no_compute_processes"}:
            raise ValueError("Config requires exactly: " + ", ".join(sorted(required)))
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("Only schema_version 1 is supported")
        if values["dataset_version"] != "1.3.1":
            raise ValueError("Formal evaluation requires Virtual KITTI 1.3.1")
        if values["camera"] != "monocular":
            raise ValueError("1.3.1 has one monocular camera; Camera_0/Camera_1 are unsupported")
        paths = {}
        for name in ("raw_root", "archive_root", "prepared_root", "sequences_file"):
            if not isinstance(values[name], str) or not values[name].strip():
                raise ValueError(f"{name} must be a nonempty path")
            paths[name] = (path.parent / values[name]).resolve()
        for name in ("raw_root", "archive_root"):
            a, b = paths["prepared_root"], paths[name]
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("prepared_root must not overlap dataset source paths")
        from .gpu import gpu_policy
        gpu_policy(values)
        from .backends import normalize_model_key, resolve_config
        from .backends.common import freeze_json
        raw_models = values.get("models", {})
        if not isinstance(raw_models, dict):
            raise ValueError("models must be a mapping")
        models = {}
        for key, model in raw_models.items():
            normalized = normalize_model_key(key)
            if key != normalized or normalized in models:
                raise ValueError("models must use unique canonical six-model keys")
            if not isinstance(model, dict):
                raise ValueError("model configuration must be a mapping")
            model = dict(model)
            for name in ("interpreter", "project_root", "checkpoint", "dependency_path",
                         "salad_checkpoint", "dino_checkpoint", "torch_home"):
                if name in model:
                    if not isinstance(model[name], str) or not model[name]:
                        raise ValueError(f"invalid {name}")
                    model[name] = str((path.parent / model[name]).absolute())
            models[normalized] = resolve_config(normalized, model)
        return VirtualKittiConfig(1, sequence_ids=read_sequence_ids(paths["sequences_file"]), models=freeze_json(models), **paths)
    except (OSError, ValueError, TypeError) as exc:
        raise DatasetValidationError("INVALID_CONFIG", str(exc)) from exc
