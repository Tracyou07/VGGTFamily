"""Strict, filesystem-resolved configuration for KITTI Odometry."""
from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
import json
from pathlib import Path
import re

PROTOCOL_ID = "kitti-odometry-image2-c2w-v1"

class DatasetValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

@dataclass(frozen=True)
class KittiConfig:
    schema_version: int
    raw_root: Path
    archive_root: Path
    prepared_root: Path
    sequences_file: Path
    sequence_ids: tuple[str, ...]
    color_root: Path | None = None
    aux_root: Path | None = None
    models: Mapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if self.color_root is None:
            object.__setattr__(self, "color_root", self.raw_root)
        if self.aux_root is None:
            object.__setattr__(self, "aux_root", self.raw_root)

def read_sequence_ids(path: Path) -> tuple[str, ...]:
    try:
        ids = tuple(line.strip() for line in Path(path).read_text().splitlines() if line.strip())
    except OSError as exc:
        raise DatasetValidationError("INVALID_SEQUENCE", str(exc)) from exc
    if not ids or any(re.fullmatch(r"[0-9]{2}", s) is None for s in ids) or tuple(sorted(set(ids))) != ids:
        raise DatasetValidationError("INVALID_SEQUENCE", "Sequence IDs must be unique, ascending two-digit ASCII IDs")
    return ids

def load_config(path: Path) -> KittiConfig:
    path = Path(path).resolve()
    try:
        values = json.loads(path.read_text())
        expected = {"schema_version", "raw_root", "archive_root", "prepared_root", "sequences_file"}
        optional = {"color_root", "aux_root", "models", "gpu_min_free_mib", "gpu_require_no_compute_processes"}
        if not isinstance(values, dict) or not expected.issubset(values) or set(values) - expected - optional:
            raise ValueError("Config requires exactly: " + ", ".join(sorted(expected)))
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("Only schema_version 1 is supported")
        paths = {}
        for field in expected - {"schema_version"}:
            value = values[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty path string")
            paths[field] = (path.parent / value).resolve()
        for source in ("raw_root", "archive_root"):
            a, b = paths["prepared_root"], paths[source]
            if a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("prepared_root must not overlap raw_root or archive_root")
        data_roots = {}
        for field in ("color_root", "aux_root"):
            value = values.get(field, values["raw_root"])
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty path string")
            resolved = (path.parent / value).resolve()
            if not resolved.is_relative_to(paths["raw_root"]):
                raise ValueError(f"{field} must remain under raw_root")
            data_roots[field] = resolved
        ids = read_sequence_ids(paths["sequences_file"])
        if any(int(s) > 10 for s in ids):
            raise ValueError("Official pose evaluation supports sequences 00 through 10")
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
        return KittiConfig(schema_version=1, sequence_ids=ids, models=freeze_json(models),
                           **paths, **data_roots)
    except (OSError, ValueError, TypeError) as exc:
        raise DatasetValidationError("INVALID_CONFIG", str(exc)) from exc
