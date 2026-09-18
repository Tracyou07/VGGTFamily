"""Immutable run inputs and live, content-based provenance for resume/aggregation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from typing import Mapping, Union
from types import MappingProxyType

from .data import PreparedSequence, verify_prepared_sequence
from .config import VirtualKittiConfig, validate_sequence_ids
from .io import canonical_json, content_sha256, sha256_file
from .metrics import validate_frame_ids

JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]
SOURCE_SUFFIXES = {".py", ".pyi", ".sh", ".bash", ".json", ".yaml", ".yml", ".toml",
                   ".cfg", ".ini", ".txt", ".cpp", ".c", ".cu", ".h", ".hpp", ".so"}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
                 "results", "outputs", "build", "dist", ".superpowers"}


def _plain(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("PROVENANCE_INVALID: JSON keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    canonical_json(value)
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _snapshot_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    # Immutable bytes own the storage: even setflags(write=True) cannot reopen it.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _snapshot_prepared(prepared: PreparedSequence) -> PreparedSequence:
    return PreparedSequence(
        sequence=prepared.sequence,
        frame_ids=tuple(prepared.frame_ids),
        image_paths=tuple(Path(path) for path in prepared.image_paths),
        timestamps_s=None,
        poses_c2w=_snapshot_array(prepared.poses_c2w),
        intrinsics=_snapshot_array(prepared.intrinsics),
        manifest_path=Path(prepared.manifest_path),
        manifest_sha256=prepared.manifest_sha256,
    )


@dataclass(frozen=True)
class RunPlan:
    config_path: Path
    config_payload: Mapping[str, JSONValue]
    model_key: str
    model_config: Mapping[str, JSONValue]
    sequence_ids: tuple[str, ...]
    prepared_sequences: tuple[PreparedSequence, ...]
    output_dir: Path
    device: str
    timeout_s: float
    command: tuple[str, ...]
    repository_root: Path
    metric_protocol_id: str
    preflight_provenance_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.config_payload, Mapping) or not isinstance(self.model_config, Mapping):
            raise ValueError("PROVENANCE_INVALID: config mappings required")
        for name in ("sequence_ids", "command"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "prepared_sequences",
                           tuple(_snapshot_prepared(p) for p in self.prepared_sequences))
        for name in ("config_path", "output_dir", "repository_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "config_payload", _freeze(_plain(self.config_payload)))
        object.__setattr__(self, "model_config", _freeze(_plain(self.model_config)))


def _read(path: Path):
    # Import lazily to keep the artifact/provenance boundary acyclic.
    from .results import read_json
    return read_json(path)


def source_fingerprint(root: Path, excluded_paths=()) -> str:
    root = Path(root).resolve()
    excluded = tuple(Path(p).resolve() for p in excluded_paths)
    if not root.is_dir() or any(root.is_relative_to(p) for p in excluded):
        raise ValueError(f"SOURCE_MISSING_OR_EXCLUDED: {root}")
    files = []
    for directory, names, filenames in os.walk(root):
        current = Path(directory)
        names[:] = sorted(n for n in names if n not in IGNORED_PARTS
                          and not any((current / n).resolve().is_relative_to(p) for p in excluded))
        for name in sorted(filenames):
            path = current / name
            if path.suffix.lower() in SOURCE_SUFFIXES or name in {"Dockerfile", "Makefile"}:
                files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    if not files:
        raise ValueError(f"SOURCE_EMPTY: {root}")
    return content_sha256(files)


def _file(path: Path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _interpreter(path: Path):
    # Preserve the selected environment's executable path, even if it is a symlink.
    path = Path(path).absolute()
    try:
        completed = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                                   check=True, timeout=15)
    except subprocess.SubprocessError as exc:
        raise ValueError(f"PROVENANCE_INTERPRETER: {exc}") from exc
    version = (completed.stdout or completed.stderr).strip()
    if not version.startswith("Python "):
        raise ValueError("PROVENANCE_INVALID: interpreter did not identify as Python")
    return {"path": str(path), "version": version, "sha256": sha256_file(path)}


def _check_manifest(record, config):
    _read(Path(record["manifest"]["path"]))  # Reject duplicate keys and nonfinite JSON.
    dataset = VirtualKittiConfig(1, Path(config["raw_root"]), Path(config["archive_root"]),
                          Path(config["prepared_root"]), Path(config["sequences_file"]),
                          tuple(config["sequence_ids"]), dataset_version=config["dataset_version"], camera=config["camera"])
    verified = verify_prepared_sequence(dataset, record["sequence"])
    if (str(verified.manifest_path.resolve()) != record["manifest"]["path"]
        or verified.manifest_sha256 != record["manifest"]["sha256"]):
        raise ValueError("SOURCE_MANIFEST_MISMATCH")
    if list(verified.frame_ids) != record["frame_ids"]:
        raise ValueError("FRAME_ID_MANIFEST_MISMATCH")
    if content_sha256(record["frame_ids"]) != record["frame_ids_sha256"]:
        raise ValueError("FRAME_ID_HASH_MISMATCH")
    return verified


def _auxiliary_paths(model):
    """Only configured native inputs, never unrelated shared Torch cache entries."""
    paths = {}
    for name in ("salad_checkpoint", "dino_checkpoint"):
        if name in model:
            paths[name] = ("file", Path(model[name]).resolve())
    if "dependency_path" in model:
        paths["dependency_path"] = ("tree", Path(model["dependency_path"]).resolve())
    if "torch_home" in model:
        root = Path(model["torch_home"]).resolve() / "hub"
        paths["torch_dinov2_source"] = ("tree", root / "facebookresearch_dinov2_main")
        paths["torch_salad_checkpoint"] = ("file", root / "checkpoints/dino_salad.ckpt")
    return paths


def model_provenance_paths(model):
    """All model-side paths bound by provenance, including complete source trees."""
    return (Path(model["project_root"]), Path(model["checkpoint"]), Path(model["interpreter"]),
            *(path for _, path in _auxiliary_paths(model).values()))


def _auxiliary_sources(model):
    return {name: {"kind": kind, "path": str(path),
                   "sha256": source_fingerprint(path) if kind == "tree" else sha256_file(path)}
            for name, (kind, path) in _auxiliary_paths(model).items()}


def build_provenance(plan: RunPlan) -> dict[str, JSONValue]:
    config, model = _plain(plan.config_payload), _plain(plan.model_config)
    ids = validate_sequence_ids(plan.sequence_ids)
    if tuple(p.sequence for p in plan.prepared_sequences) != ids:
        raise ValueError("PROVENANCE_INVALID: prepared sequence order mismatch")
    if not isinstance(plan.timeout_s, (int, float)) or isinstance(plan.timeout_s, bool) or plan.timeout_s <= 0:
        raise ValueError("PROVENANCE_INVALID: timeout must be positive")
    for value in (plan.model_key, plan.device, plan.metric_protocol_id):
        if not isinstance(value, str) or not value:
            raise ValueError("PROVENANCE_INVALID: nonempty identifiers required")
    if not plan.command or any(not isinstance(v, str) or not v for v in plan.command):
        raise ValueError("PROVENANCE_INVALID: command must contain nonempty strings")
    sequences = {}
    for prepared in plan.prepared_sequences:
        frames = list(validate_frame_ids(prepared.frame_ids))
        sequences[prepared.sequence] = {"sequence": prepared.sequence, "frame_ids": frames,
            "frame_ids_sha256": content_sha256(frames),
            "manifest": {"path": str(prepared.manifest_path.resolve()), "sha256": prepared.manifest_sha256}}
        verified = _check_manifest(sequences[prepared.sequence], config)
        if (prepared.image_paths != verified.image_paths
            or any(not np.array_equal(getattr(prepared, name), getattr(verified, name))
                   for name in ("poses_c2w", "intrinsics"))):
            raise ValueError("PREPARED_VALUES_MISMATCH")
    excluded = sorted({str(Path(p).resolve()) for p in
                       (plan.output_dir, config["prepared_root"], config["raw_root"], config["archive_root"])})
    def tree(path):
        return {"path": str(Path(path).resolve()), "excluded_paths": excluded,
                "sha256": source_fingerprint(Path(path), excluded)}
    payload = {"schema_version": 1, "sequence_file": _file(Path(config["sequences_file"])), "config_file": _file(plan.config_path), "config_payload": config,
        "model_key": plan.model_key, "model_config": model, "sequence_ids": list(ids), "sequences": sequences,
        "output_dir": str(plan.output_dir.resolve()), "device": plan.device, "timeout_s": plan.timeout_s,
        "command": list(plan.command), "metric_protocol_id": plan.metric_protocol_id,
        "model_source": tree(model["project_root"]),
        "checkpoint": _file(Path(model["checkpoint"])), "interpreter": _interpreter(Path(model["interpreter"])),
        "package_source": tree(plan.repository_root), "auxiliary_sources": _auxiliary_sources(model)}
    payload["provenance_id"] = content_sha256(payload)
    if plan.preflight_provenance_id is not None and payload["provenance_id"] != plan.preflight_provenance_id:
        raise ValueError("SOURCE_CHANGED: run inputs differ from the immutable preflight snapshot")
    return payload


def validate_provenance(payload, *, check_sources: bool = True) -> None:
    try:
        if not isinstance(payload, dict):
            raise ValueError("expected object")
        data = dict(payload)
        digest = data.pop("provenance_id")
        fields = {"schema_version", "sequence_file", "config_file", "config_payload", "model_key", "model_config",
                  "sequence_ids", "sequences", "output_dir", "device", "timeout_s", "command",
                  "metric_protocol_id", "model_source", "checkpoint", "interpreter", "package_source", "auxiliary_sources"}
        if set(data) != fields or type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported schema")
        if content_sha256(data) != digest:
            raise ValueError("content hash mismatch")
        if type(data["timeout_s"]) not in (int, float) or not math.isfinite(data["timeout_s"]) or data["timeout_s"] <= 0:
            raise ValueError("invalid timeout")
        for name in ("model_key", "device", "metric_protocol_id", "output_dir"):
            if not isinstance(data[name], str) or not data[name]:
                raise ValueError(f"invalid {name}")
        if not isinstance(data["command"], list) or not data["command"] or any(not isinstance(v, str) or not v for v in data["command"]):
            raise ValueError("invalid command")
        if not isinstance(data["config_payload"], dict) or not isinstance(data["model_config"], dict):
            raise ValueError("invalid configs")
        ids = validate_sequence_ids(data["sequence_ids"])
        if set(ids) != set(data["sequences"]):
            raise ValueError("sequence inventory mismatch")
        for sequence, record in data["sequences"].items():
            if record["sequence"] != sequence:
                raise ValueError("sequence mismatch")
            validate_frame_ids(record["frame_ids"])
            if record["frame_ids_sha256"] != content_sha256(record["frame_ids"]):
                raise ValueError("frame hash mismatch")
        config, model = data["config_payload"], data["model_config"]
        for field in ("raw_root", "archive_root", "prepared_root", "sequences_file"):
            if not isinstance(config[field], str) or not Path(config[field]).is_absolute():
                raise ValueError("normalized configuration requires absolute paths")
        if not set(ids).issubset(validate_frame_ids(config["sequence_ids"])):
            raise ValueError("config/sequence mismatch")
        expected_excluded = sorted({str(Path(p).resolve()) for p in
            (data["output_dir"], config["prepared_root"], config["raw_root"], config["archive_root"])})
        if (data["model_source"]["path"] != str(Path(model["project_root"]).resolve())
            or data["checkpoint"]["path"] != str(Path(model["checkpoint"]).resolve())
            or data["interpreter"]["path"] != str(Path(model["interpreter"]).absolute())
            or data["sequence_file"]["path"] != str(Path(config["sequences_file"]).resolve())):
            raise ValueError("source/config path mismatch")
        for name in ("model_source", "package_source"):
            if data[name]["excluded_paths"] != expected_excluded:
                raise ValueError("source exclusions mismatch")
        auxiliary = data["auxiliary_sources"]
        paths = _auxiliary_paths(model)
        if not isinstance(auxiliary, dict) or set(auxiliary) != set(paths):
            raise ValueError("auxiliary source inventory mismatch")
        for name, (kind, path) in paths.items():
            record = auxiliary[name]
            if (set(record) != {"kind", "path", "sha256"} or record["kind"] != kind
                or record["path"] != str(path)):
                raise ValueError("auxiliary source/config mismatch")
        if check_sources:
            if _auxiliary_sources(model) != auxiliary:
                raise ValueError("SOURCE_CHANGED: auxiliary native assets")
            for name in ("config_file", "checkpoint", "sequence_file"):
                record = data[name]
                if sha256_file(Path(record["path"])) != record["sha256"]:
                    raise ValueError(f"SOURCE_CHANGED: {name}")
            for name in ("model_source", "package_source"):
                record = data[name]
                if source_fingerprint(Path(record["path"]), record["excluded_paths"]) != record["sha256"]:
                    raise ValueError(f"SOURCE_CHANGED: {name}")
            if _interpreter(Path(data["interpreter"]["path"])) != data["interpreter"]:
                raise ValueError("SOURCE_CHANGED: interpreter")
            for record in data["sequences"].values():
                _check_manifest(record, data["config_payload"])
    except (OSError, KeyError, TypeError, ValueError, AttributeError, OverflowError, subprocess.SubprocessError) as exc:
        raise ValueError(f"PROVENANCE_INVALID: {exc}") from exc
