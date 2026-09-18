"""Validated ScanNet model execution with provenance-bound resumability."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .data import SceneData, load_scene

METRIC_KEYS = (
    "chamfer_distance",
    "ate",
    "are",
    "rpe_rot",
    "rpe_trans",
    "inference_time_ms",
)
SCENE_METRIC_KEYS = METRIC_KEYS + (
    "scale_factor",
    "aligned_chamfer_distance",
    "aligned_ate",
    "aligned_are",
    "aligned_rpe_rot",
    "aligned_rpe_trans",
    "aligned_scale_factor",
)
RUN_SCHEMA_VERSION = 1
_CODE_SUFFIXES = {
    ".py",
    ".pyi",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".ini",
    ".sh",
}
_EXCLUDED_PARTS = {
    ".git",
    ".runtime",
    "__pycache__",
    "build",
    "datasets",
    "dist",
    "logs",
    "output",
    "outputs",
    "pretrained",
    "result",
    "results",
    "weights",
    "checkpoints",
}


class RunPreflightError(ValueError):
    pass


class OutputConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPlan:
    config_path: Path
    config: dict[str, Any]
    model_name: str
    model_config: dict[str, Any]
    prepared_root: Path
    scenes: tuple[SceneData, ...]
    output_dir: Path
    max_frames: int
    device: str
    overrides: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value: {value}")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(_jsonable(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def load_config(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunPreflightError(f"invalid config: {config_path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RunPreflightError("config must be a schema_version=1 JSON object")
    if not isinstance(value.get("models"), dict) or not value["models"]:
        raise RunPreflightError("config models must be a non-empty object")
    return config_path, value


def _scene_ids(values: Iterable[str]) -> tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise RunPreflightError("scene ids must be non-empty strings")
        if value in seen:
            raise RunPreflightError(f"duplicate scene id: {value}")
        seen.add(value)
        result.append(value)
    if not result:
        raise RunPreflightError("at least one scene id is required")
    return tuple(result)


def _override_gt(scene: SceneData, root: Path) -> SceneData:
    candidates = (root / scene.scene_id / scene.gt_ply.name, root / scene.gt_ply.name)
    target = next(
        (candidate for candidate in candidates if candidate.is_file()), candidates[0]
    )
    if not target.is_file():
        raise RunPreflightError(f"missing GT override for {scene.scene_id}: {target}")
    if _sha256(target) != _sha256(scene.gt_ply):
        raise RunPreflightError(
            f"GT override hash mismatch for {scene.scene_id}: {target}"
        )
    return replace(scene, gt_ply=target.resolve())


def preflight_inputs(
    config_path,
    model_name,
    output_dir,
    scene_ids,
    *,
    max_frames=1000,
    device="cuda",
    prepared_root=None,
    gt_ply_dir=None,
    backend_overrides=None,
    chamfer_max_dist=0.5,
    verify_input_hashes=False,
) -> RunPlan:
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames <= 0
    ):
        raise RunPreflightError("max_frames must be a positive integer")
    if not isinstance(device, str) or not device:
        raise RunPreflightError("device must be a non-empty string")
    if not isinstance(verify_input_hashes, bool):
        raise RunPreflightError("verify_input_hashes must be a boolean")
    if isinstance(chamfer_max_dist, bool):
        raise RunPreflightError("chamfer_max_dist must be finite and positive")
    try:
        chamfer_value = float(chamfer_max_dist)
    except (TypeError, ValueError) as error:
        raise RunPreflightError(
            "chamfer_max_dist must be finite and positive"
        ) from error
    if not math.isfinite(chamfer_value) or chamfer_value <= 0:
        raise RunPreflightError("chamfer_max_dist must be finite and positive")
    ids = _scene_ids(scene_ids)
    path, config = load_config(config_path)
    if model_name not in config["models"]:
        raise RunPreflightError(f"unknown configured model: {model_name}")
    model_config = dict(config["models"][model_name])
    overrides = {} if backend_overrides is None else dict(backend_overrides)
    unknown = set(overrides) - set(model_config)
    if unknown:
        raise RunPreflightError(
            f"backend overrides do not match configured keys: {sorted(unknown)}"
        )
    model_config.update(overrides)
    root = Path(prepared_root or config.get("prepared_root", "")).expanduser().resolve()
    if not root.is_dir():
        raise RunPreflightError(f"prepared root does not exist: {root}")
    gt_root = (
        Path(gt_ply_dir).expanduser().resolve() if gt_ply_dir is not None else None
    )
    from .fastvggt_eval import validate_scene_inputs

    scenes = []
    errors = []
    for scene_id in ids:
        try:
            scene = load_scene(
                root,
                scene_id,
                max_frames=max_frames,
                verify_hashes=verify_input_hashes,
            )
            if gt_root is not None:
                scene = _override_gt(scene, gt_root)
            validate_scene_inputs(scene)
            scenes.append(scene)
        except Exception as error:
            errors.append(f"{scene_id}: {error}")
    if errors:
        raise RunPreflightError(
            "requested scene preflight failed: " + "; ".join(errors)
        )
    for key in ("python", "project_root", "checkpoint"):
        if not isinstance(model_config.get(key), str) or not model_config[key]:
            errors.append(f"model {model_name} missing {key}")
    for key in ("python", "project_root", "checkpoint"):
        if isinstance(model_config.get(key), str):
            resolved = Path(model_config[key]).expanduser().resolve()
            if key == "project_root" and not resolved.is_dir():
                errors.append(f"model project_root does not exist: {resolved}")
            elif key != "project_root" and not resolved.is_file():
                errors.append(f"model {key} does not exist: {resolved}")
            model_config[key] = str(resolved)
    if errors:
        raise RunPreflightError("model preflight failed: " + "; ".join(errors))
    recorded = {
        "prepared_root": str(root),
        "gt_ply_dir": str(gt_root) if gt_root else None,
        "backend": _jsonable(overrides),
        "verify_input_hashes": verify_input_hashes,
    }
    return RunPlan(
        path,
        config,
        model_name,
        model_config,
        root,
        tuple(scenes),
        Path(output_dir).expanduser().resolve(),
        max_frames,
        device,
        recorded,
    )


def _source_fingerprint(root: Path) -> dict[str, Any]:
    files = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = sorted(
            name
            for name in names
            if name not in _EXCLUDED_PARTS and not name.startswith(".")
        )
        base = Path(directory)
        for filename in sorted(filenames):
            path = base / filename
            if path.suffix.lower() in _CODE_SUFFIXES and path.is_file():
                files.append(path)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    if not files:
        raise RunPreflightError(
            f"model source fingerprint found no code/config files: {root}"
        )
    return {"root": str(root), "sha256": digest.hexdigest(), "file_count": len(files)}


def _runtime_fingerprint(repository_root: Path) -> dict[str, Any]:
    repository_root = Path(repository_root).resolve()
    package_root = repository_root / "scannet_eval"
    files = sorted(package_root.rglob("*.py")) if package_root.is_dir() else []
    entrypoint = repository_root / "eval_scannet.py"
    if entrypoint.is_file():
        files.append(entrypoint)
    files = sorted(path for path in files if path.is_file())
    if not files:
        raise RunPreflightError(
            f"evaluation runtime fingerprint found no Python files: {repository_root}"
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(repository_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


def _git_head(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _versions() -> dict[str, str]:
    result = {}
    for name in ("numpy", "open3d", "evo", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def build_provenance(plan: RunPlan) -> dict[str, Any]:
    from . import fastvggt_eval

    protocol_id = getattr(fastvggt_eval, "PROTOCOL_ID", None)
    if protocol_id != "fastvggt_scannet_evo132":
        raise RunPreflightError(
            f"unsupported or missing evaluator protocol id: {protocol_id!r}"
        )
    repo = Path(__file__).resolve().parents[1]
    protocol_files = [
        repo / "scannet_eval" / "fastvggt_eval.py",
        repo / "scannet_eval" / "vendor" / "fastvggt_eval_utils.py",
        repo / "reference" / "FastVGGT" / "SOURCE.json",
    ]
    protocol_digest = hashlib.sha256()
    for path in protocol_files:
        if not path.is_file():
            raise RunPreflightError(f"missing evaluator protocol source: {path}")
        protocol_digest.update(path.relative_to(repo).as_posix().encode())
        protocol_digest.update(bytes.fromhex(_sha256(path)))
    source = Path(plan.model_config["project_root"])
    checkpoint = Path(plan.model_config["checkpoint"])
    config_without_runtime = {
        key: value
        for key, value in plan.model_config.items()
        if key not in {"python", "project_root", "checkpoint"}
    }
    auxiliary_assets = {}
    dependency_fingerprints = {}
    if plan.model_name == "long":
        for key in ("salad_checkpoint", "dino_checkpoint"):
            candidate = (
                Path(plan.model_config[key]) if plan.model_config.get(key) else None
            )
            if candidate is not None and candidate.is_file():
                auxiliary_assets[key] = {
                    "path": str(candidate),
                    "sha256": _sha256(candidate),
                    "size": candidate.stat().st_size,
                }
        dependency = Path(plan.model_config["dependency_path"])
        if dependency.is_dir():
            dependency_fingerprints["dependency_path"] = _source_fingerprint(dependency)
    elif plan.model_name == "slam" and plan.model_config.get("torch_home"):
        torch_home = Path(plan.model_config["torch_home"])
        salad = torch_home / "hub" / "checkpoints" / "dino_salad.ckpt"
        dino_source = torch_home / "hub" / "facebookresearch_dinov2_main"
        if salad.is_file():
            auxiliary_assets["salad_checkpoint"] = {
                "path": str(salad),
                "sha256": _sha256(salad),
                "size": salad.stat().st_size,
            }
        if dino_source.is_dir():
            dependency_fingerprints["dino_hub_source"] = _source_fingerprint(
                dino_source
            )
    return _jsonable(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "protocol": {
                "id": protocol_id,
                "source_sha256": protocol_digest.hexdigest(),
                "scene_metric_keys": list(SCENE_METRIC_KEYS),
                "aggregate_metric_keys": list(METRIC_KEYS),
            },
            "model": {
                "name": plan.model_name,
                "config": config_without_runtime,
                "configured_paths": {
                    key: plan.config["models"][plan.model_name].get(key)
                    for key in ("python", "project_root", "checkpoint")
                },
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _sha256(checkpoint),
                    "size": checkpoint.stat().st_size,
                },
                "auxiliary_assets": auxiliary_assets,
                "dependency_fingerprints": dependency_fingerprints,
                "source_fingerprint": _source_fingerprint(source),
                "source_git_head": _git_head(source),
            },
            "data": {"prepared_root": str(plan.prepared_root)},
            "runtime": {"source_fingerprint": _runtime_fingerprint(repo)},
            "scenes": [
                {
                    "scene_id": scene.scene_id,
                    "manifest_sha256": scene.manifest_sha256,
                    "selected_frame_ids": list(scene.frame_ids),
                    "gt_ply": str(scene.gt_ply),
                    "gt_sha256": _sha256(scene.gt_ply),
                }
                for scene in plan.scenes
            ],
            "request": {
                "scene_ids": [scene.scene_id for scene in plan.scenes],
                "max_frames": plan.max_frames,
                "device": plan.device,
                "output_dir": str(plan.output_dir),
            },
            "overrides": plan.overrides,
            "config": {
                "path": str(plan.config_path),
                "sha256": _sha256(plan.config_path),
            },
            "environment": {
                "interpreter": str(Path(sys.executable).resolve()),
                "platform": platform.platform(),
                "variables": {
                    key: os.environ.get(key)
                    for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "TORCH_HOME")
                },
                "versions": _versions(),
            },
            "git_head": _git_head(repo),
        }
    )


def _provenance_id(provenance):
    return hashlib.sha256(
        json.dumps(
            provenance, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise OutputConflictError(f"invalid JSON artifact {path}: {error}") from error


def _prepare_output(output: Path, provenance: dict[str, Any], resume: bool) -> str:
    manifest = output / "run_manifest.json"
    provenance_id = _provenance_id(provenance)
    if output.exists() and any(output.iterdir()):
        if not manifest.is_file():
            raise OutputConflictError(
                f"unknown existing output cannot be overwritten: {output}"
            )
        existing = _read_json(manifest)
        if existing != provenance:
            raise OutputConflictError(
                f"existing output provenance does not match requested run: {output}"
            )
        if not resume:
            raise OutputConflictError(f"existing output requires --resume: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(manifest, provenance)
    return provenance_id


def _validated_scene_metrics(metrics, *, context, error_type):
    if not isinstance(metrics, dict) or set(metrics) != set(SCENE_METRIC_KEYS):
        raise error_type(
            f"{context} must contain exactly the 13 FastVGGT per-scene metric keys"
        )
    normalized = {}
    for key in SCENE_METRIC_KEYS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise error_type(f"{context} has a non-numeric value for {key}")
        normalized[key] = float(value)
        if not math.isfinite(normalized[key]):
            raise error_type(f"{context} has a non-finite value for {key}")
    return normalized


def _result(path: Path, scene_id: str, provenance_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        if path.parent.exists():
            raise OutputConflictError(
                f"incomplete scene output is missing result.json: {path.parent}"
            )
        return None
    value = _read_json(path)
    if (
        not isinstance(value, dict)
        or value.get("scene_id") != scene_id
        or value.get("provenance_sha256") != provenance_id
    ):
        raise OutputConflictError(f"stale scene result cannot be resumed: {path}")
    metrics = _validated_scene_metrics(
        value.get("metrics"),
        context=f"result metrics in {path}",
        error_type=OutputConflictError,
    )
    metrics_path = path.parent / "metrics.json"
    if not metrics_path.is_file():
        raise OutputConflictError(f"missing per-scene metrics artifact: {metrics_path}")
    metrics_artifact = _validated_scene_metrics(
        _read_json(metrics_path),
        context=f"metrics artifact {metrics_path}",
        error_type=OutputConflictError,
    )
    if metrics_artifact != metrics:
        raise OutputConflictError(
            f"metrics artifact does not match result.json: {metrics_path}"
        )
    _jsonable(value)
    value["metrics"] = metrics
    return value


def _summary(
    output: Path, provenance: dict[str, Any], provenance_id: str
) -> dict[str, Any]:
    ids = provenance["request"]["scene_ids"]
    records = {}
    failures = {}
    for scene_id in ids:
        result = _result(output / scene_id / "result.json", scene_id, provenance_id)
        if result is not None:
            records[scene_id] = result
        failure = output / "failures" / f"{scene_id}.json"
        if result is None and failure.is_file():
            failures[scene_id] = _read_json(failure)
    averages = (
        {
            key: float(np.mean([record["metrics"][key] for record in records.values()]))
            for key in METRIC_KEYS
        }
        if records
        else {}
    )
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "provenance_sha256": provenance_id,
        "expected_count": len(ids),
        "success_count": len(records),
        "failed_count": len(ids) - len(records),
        "complete": len(records) == len(ids) and bool(ids),
        "successful_scenes": list(records),
        "failed_scenes": [scene for scene in ids if scene not in records],
        "average_metrics": averages,
        "provenance": provenance,
    }
    _atomic_json(
        output / "all_scenes_metrics.json",
        {scene: record["metrics"] for scene, record in records.items()},
    )
    _atomic_json(output / "average_metrics.json", averages)
    _atomic_json(output / "summary.json", summary)
    return summary


def aggregate_run(output_dir) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    manifest = output / "run_manifest.json"
    if not manifest.is_file():
        raise RunPreflightError(f"missing run manifest: {manifest}")
    provenance = _read_json(manifest)
    ids = provenance.get("request", {}).get("scene_ids")
    if not isinstance(ids, list) or not ids:
        raise RunPreflightError("run manifest has zero requested scenes")
    summary = _summary(output, provenance, _provenance_id(provenance))
    if summary["success_count"] == 0:
        raise RunPreflightError("cannot aggregate zero successful scenes")
    return summary


def run_evaluation(
    config_path,
    model_name,
    output_dir,
    scene_ids,
    *,
    max_frames=1000,
    device="cuda",
    resume=False,
    prepared_root=None,
    gt_ply_dir=None,
    backend_overrides=None,
    backend_factory: Callable[..., Any] | None = None,
    evaluator: Callable[..., Any] | None = None,
    chamfer_max_dist=0.5,
    plot=False,
    verify_input_hashes=False,
) -> dict[str, Any]:
    plan = preflight_inputs(
        config_path,
        model_name,
        output_dir,
        scene_ids,
        max_frames=max_frames,
        device=device,
        prepared_root=prepared_root,
        gt_ply_dir=gt_ply_dir,
        backend_overrides=backend_overrides,
        chamfer_max_dist=chamfer_max_dist,
        verify_input_hashes=verify_input_hashes,
    )
    if backend_factory is None:
        from .backends import create_backend, resolve_config

        plan = replace(
            plan, model_config=resolve_config(plan.model_name, plan.model_config)
        )
        backend_factory = create_backend
    provenance = build_provenance(plan)
    provenance["overrides"]["chamfer_max_dist"] = float(chamfer_max_dist)
    provenance["overrides"]["plot"] = bool(plot)
    provenance = _jsonable(provenance)
    provenance_id = _prepare_output(plan.output_dir, provenance, bool(resume))
    pending = []
    for scene in plan.scenes:
        if (
            _result(
                plan.output_dir / scene.scene_id / "result.json",
                scene.scene_id,
                provenance_id,
            )
            is None
        ):
            pending.append(scene)
    if not pending:
        return _summary(plan.output_dir, provenance, provenance_id)
    if evaluator is None:
        from .fastvggt_eval import evaluate_prediction

        evaluator = evaluate_prediction
    try:
        backend = backend_factory(plan.model_name, plan.model_config, plan.device)
    except Exception as error:
        for scene in pending:
            _atomic_json(
                plan.output_dir / "failures" / f"{scene.scene_id}.json",
                {
                    "scene_id": scene.scene_id,
                    "stage": "backend_factory",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "provenance_sha256": provenance_id,
                },
            )
        return _summary(plan.output_dir, provenance, provenance_id)
    for scene in pending:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{scene.scene_id}.", suffix=".partial", dir=plan.output_dir
            )
        )
        try:
            prediction = backend.predict(scene, staging)
            metrics = evaluator(
                scene,
                prediction,
                staging,
                chamfer_max_dist=float(chamfer_max_dist),
                plot=bool(plot),
            )
            metrics = _validated_scene_metrics(
                metrics, context="evaluator metrics", error_type=ValueError
            )
            prediction_record = {
                "inference_seconds": getattr(prediction, "inference_seconds"),
                "peak_allocated_bytes": getattr(prediction, "peak_allocated_bytes"),
                "peak_reserved_bytes": getattr(prediction, "peak_reserved_bytes"),
                "metadata": getattr(prediction, "metadata"),
            }
            record = _jsonable(
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "scene_id": scene.scene_id,
                    "provenance_sha256": provenance_id,
                    "frame_ids": list(scene.frame_ids),
                    "metrics": metrics,
                    "prediction": prediction_record,
                }
            )
            _atomic_json(staging / "metrics.json", metrics)
            _atomic_json(staging / "result.json", record)
            destination = plan.output_dir / scene.scene_id
            if destination.exists():
                raise OutputConflictError(
                    f"scene output already exists without a valid result: {destination}"
                )
            os.replace(staging, destination)
            try:
                (plan.output_dir / "failures" / f"{scene.scene_id}.json").unlink()
            except FileNotFoundError:
                pass
        except Exception as error:
            shutil.rmtree(staging, ignore_errors=True)
            safe_error = str(error)
            _atomic_json(
                plan.output_dir / "failures" / f"{scene.scene_id}.json",
                {
                    "scene_id": scene.scene_id,
                    "stage": "scene",
                    "error_type": type(error).__name__,
                    "error": safe_error,
                    "provenance_sha256": provenance_id,
                },
            )
    return _summary(plan.output_dir, provenance, provenance_id)


__all__ = [
    "METRIC_KEYS",
    "SCENE_METRIC_KEYS",
    "OutputConflictError",
    "RunPlan",
    "RunPreflightError",
    "aggregate_run",
    "build_provenance",
    "load_config",
    "preflight_inputs",
    "run_evaluation",
]
