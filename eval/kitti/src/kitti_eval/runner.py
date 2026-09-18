"""CPU preflight, isolated sequence execution and exact provenance-bound resume."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Sequence
import uuid

import numpy as np

from .backends import doctor_backend, normalize_model_key, required_assets
from .backends.common import BackendRequest, validate_prediction
from .backend_worker import WorkerResult, launch_worker
from .gpu import admit_gpu as _validate_device
from .config import load_config
from .data import PreparedSequence, verify_prepared_sequence
from .metrics import PROTOCOL_ID, ate_rmse_m
from .provenance import RunPlan, build_provenance, model_provenance_paths
from .results import (AggregateSummary, aggregate, atomic_write_json, load_result_pair,
                      metrics_record, read_json, write_result_pair)

MIN_FREE_BYTES = 256 * 2**20
ROOT_ARTIFACTS = {"run_manifest.json", "all_sequences_metrics.json",
                  "average_metrics.json", "summary.json"}
SEQUENCE_PATTERN = re.compile(r"(?:0[0-9]|10)")


def _execution_environment():
    return {key: os.environ.get(key) for key in
            ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}


def _validate_output(output, config, model, config_path):
    # A custom output below this evaluator is supported, but no input can be hidden
    # beneath the output exclusion or overwritten by result artifacts.
    protected = [config.raw_root, config.color_root, config.aux_root,
                 config.archive_root, config.prepared_root, config_path, config.sequences_file]
    files, directories, checkpoints = required_assets(model[0], model[1])
    protected += files + directories + checkpoints + list(model_provenance_paths(model[1]))
    for path in protected:
        path = Path(path).resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f"OUTPUT_COLLISION: output overlaps an input: {path}")
    if output.exists():
        if not output.is_dir():
            raise ValueError("OUTPUT_COLLISION: output is not a directory")
        for child in output.iterdir():
            if child.is_symlink():
                raise ValueError("OUTPUT_COLLISION: symlink artifact")
            if child.name == ".run.lock":
                raise ValueError("OUTPUT_LOCKED: another run or interrupted lock exists")
            if child.is_file() and child.name in ROOT_ARTIFACTS:
                continue
            if child.is_dir() and (child.name == "failures" or SEQUENCE_PATTERN.fullmatch(child.name)):
                continue
            raise ValueError(f"OUTPUT_COLLISION: unrecognized artifact {child.name}")
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
        raise ValueError("OUTPUT_UNWRITABLE")
    if shutil.disk_usage(ancestor).free < MIN_FREE_BYTES:
        raise ValueError("DISK_SPACE: at least 256 MiB free required before launch")


def preflight_run(config_path: Path, model_key: str, sequence_ids: Sequence[str],
                  output_dir: Path, device: str, *, timeout_s: float = 3600.) -> RunPlan:
    """Read-only validation of every requested sequence before a worker exists."""
    config_path, output = Path(config_path).resolve(), Path(output_dir).resolve()
    read_json(config_path)  # Strict JSON, including duplicate keys and NaN.
    config = load_config(config_path)
    key = normalize_model_key(model_key)
    ids = tuple(sequence_ids)
    if not ids or len(ids) != len(set(ids)) or any(s not in config.sequence_ids for s in ids):
        raise ValueError("INVALID_SEQUENCE: explicit unique configured sequence IDs required")
    if key not in config.models:
        raise ValueError(f"BACKEND_CONFIG: no configured model {key}")
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("TIMEOUT_INVALID: positive finite seconds required")
    model = config.models[key]
    _validate_output(output, config, (key, model), config_path)
    _validate_device(device, read_json(config_path))
    prepared = tuple(verify_prepared_sequence(config, s) for s in ids)
    # Reject a GT trajectory that cannot support the protocol before GPU work.
    for sequence in prepared:
        ate_rmse_m(sequence.frame_ids, sequence.poses_c2w, sequence.frame_ids, sequence.poses_c2w)
    status = doctor_backend(key, model)
    if not status.ready:
        raise ValueError("BACKEND_PREFLIGHT: " + "; ".join(f"{b.code}: {b.message}" for b in status.blockers))
    payload = {name: str(getattr(config, name)) for name in
               ("raw_root", "color_root", "aux_root", "archive_root", "prepared_root", "sequences_file")}
    payload.update(schema_version=config.schema_version, sequence_ids=list(config.sequence_ids),
                   execution_environment=_execution_environment())
    command = ("kitti-eval", "run", "--config", str(config_path), "--model", key,
               "--sequence", *ids, "--output", str(output), "--device", device,
               "--timeout", str(float(timeout_s)))
    plan = RunPlan(config_path, payload, key, model, ids, prepared, output, device,
                   float(timeout_s), command, Path(__file__).resolve().parents[2], PROTOCOL_ID)
    provenance = build_provenance(plan)
    return replace(plan, preflight_provenance_id=provenance["provenance_id"])


@dataclass(frozen=True)
class SequenceResult:
    sequence: str
    status: str
    result: dict
    metrics: dict | None


def _check_environment(plan):
    if dict(plan.config_payload.get("execution_environment", {})) != _execution_environment():
        raise ValueError("ENVIRONMENT_CHANGED: physical CUDA selection changed after preflight")


def run_sequence(plan: RunPlan, prepared: PreparedSequence) -> SequenceResult:
    """Compute metrics only after launch_worker has finished its measured boundary."""
    _check_environment(plan)
    provenance = build_provenance(plan)
    if prepared.sequence not in plan.sequence_ids:
        raise ValueError("INVALID_SEQUENCE: sequence absent from run plan")
    snapshot = plan.prepared_sequences[plan.sequence_ids.index(prepared.sequence)]
    if (prepared.frame_ids != snapshot.frame_ids or prepared.image_paths != snapshot.image_paths
        or prepared.manifest_path != snapshot.manifest_path
        or prepared.manifest_sha256 != snapshot.manifest_sha256
        or any(not np.array_equal(getattr(prepared, name), getattr(snapshot, name))
               for name in ("poses_c2w", "timestamps_s", "intrinsics"))):
        raise ValueError("PREPARED_MISMATCH: input differs from immutable RunPlan")
    prepared = snapshot
    directory = plan.output_dir / prepared.sequence
    if directory.exists():
        raise ValueError(f"OUTPUT_EXISTS: {directory}")
    request = BackendRequest.from_prepared(prepared, model_key=plan.model_key,
        model_config=plan.model_config, output_dir=directory / "worker",
        provenance_id=provenance["provenance_id"], device=plan.device)
    _validate_device(plan.device, read_json(plan.config_path))
    try:
        worker = launch_worker(request, plan.timeout_s)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        worker = WorkerResult("error", "BACKEND_LAUNCH_FAILED", str(exc))
    metrics = None
    try:
        if worker.status == "success":
            prediction = validate_prediction(worker.prediction, prepared.frame_ids)
            if worker.usage is None or worker.returncode != 0 or worker.signal is not None:
                raise ValueError("BACKEND_SUCCESS_INVALID: missing resources or unsuccessful exit")
            measured = ate_rmse_m(prediction.frame_ids, prediction.poses_c2w,
                                  prepared.frame_ids, prepared.poses_c2w)
            metrics = metrics_record(measured, prepared.frame_ids, prepared.sequence,
                                      plan.model_key, provenance["provenance_id"])
    except (ValueError, TypeError, KeyError) as exc:
        worker = WorkerResult("error", "METRIC_OR_PREDICTION_INVALID", str(exc),
                              returncode=worker.returncode, signal=worker.signal)
    usage = worker.usage
    result = dict(schema_version=1, model_key=plan.model_key, sequence=prepared.sequence,
        status=worker.status, input_frames=len(prepared.frame_ids),
        inference_seconds=usage.inference_seconds if usage else None,
        peak_allocated_mib=usage.peak_allocated_mib if usage else None,
        peak_reserved_mib=usage.peak_reserved_mib if usage else None,
        worker_exit_state={"returncode": worker.returncode, "signal": worker.signal},
        provenance_id=provenance["provenance_id"], metrics_sha256=None)
    write_result_pair(directory, result, metrics)
    if worker.status != "success":
        atomic_write_json(directory / "failure.json", dict(sequence=prepared.sequence,
            status=worker.status, code=worker.failure_code, message=worker.message,
            worker_exit_state=result["worker_exit_state"], provenance_id=provenance["provenance_id"]))
    return SequenceResult(prepared.sequence, worker.status, read_json(directory / "result.json"), metrics)


def _quarantine(path, failure_dir, reason):
    destination = failure_dir / f"{path.name}-{reason}-{uuid.uuid4().hex}"
    path.rename(destination)


def _fail_remaining_sequences(plan, remaining, provenance, error):
    """Commit failures against the original run identity without re-reading inputs."""
    code = str(error).split(":", 1)[0]
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", code):
        code = "RUN_PREFLIGHT_INVALIDATED"
    for prepared in remaining:
        directory = plan.output_dir / prepared.sequence
        if directory.exists():
            _quarantine(directory, plan.output_dir / "failures", "invalidated")
        result = dict(schema_version=1, model_key=plan.model_key, sequence=prepared.sequence,
            status="error", input_frames=len(prepared.frame_ids), inference_seconds=None,
            peak_allocated_mib=None, peak_reserved_mib=None,
            worker_exit_state={"returncode": None, "signal": None},
            provenance_id=provenance["provenance_id"], metrics_sha256=None)
        write_result_pair(directory, result, None)
        atomic_write_json(directory / "failure.json", dict(sequence=prepared.sequence,
            status="error", code=code, message=str(error),
            diagnostics=getattr(error, "diagnostics", {}),
            worker_exit_state=result["worker_exit_state"], provenance_id=provenance["provenance_id"]))
        aggregate(plan.output_dir)


def run_evaluation(plan: RunPlan, resume: bool = False) -> AggregateSummary:
    _check_environment(plan)
    provenance = build_provenance(plan)  # Revalidates ALL sources before the first launch.
    config = load_config(plan.config_path)
    _validate_output(plan.output_dir, config, (plan.model_key, plan.model_config), plan.config_path)
    _validate_device(plan.device, read_json(plan.config_path))
    if plan.output_dir.exists() and any(plan.output_dir.iterdir()) and not resume:
        raise ValueError("OUTPUT_EXISTS: use a new output directory or explicit --resume")
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    lock = plan.output_dir / ".run.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        failure_dir = plan.output_dir / "failures"
        failure_dir.mkdir(exist_ok=True)
        try:
            previous_matches = read_json(plan.output_dir / "run_manifest.json") == provenance
        except ValueError:
            previous_matches = False
        if not previous_matches:
            for name in ROOT_ARTIFACTS:
                path = plan.output_dir / name
                if path.exists():
                    _quarantine(path, failure_dir, "stale")
        atomic_write_json(plan.output_dir / "run_manifest.json", provenance)
        # Existing non-requested sequences are stale under this exact run selection.
        for directory in sorted(plan.output_dir.iterdir()):
            if directory.is_dir() and SEQUENCE_PATTERN.fullmatch(directory.name) and directory.name not in plan.sequence_ids:
                _quarantine(directory, failure_dir, "stale")
        for index, prepared in enumerate(plan.prepared_sequences):
            directory = plan.output_dir / prepared.sequence
            if directory.exists():
                reason = "partial" if not all((directory / n).is_file() for n in
                                              ("metrics.json", "result.json")) else "invalid"
                try:
                    if not resume or not previous_matches:
                        reason = "stale" if reason != "partial" else reason
                        raise ValueError("run manifest mismatch")
                    load_result_pair(directory, provenance)
                except ValueError:
                    _quarantine(directory, failure_dir, reason)
                else:
                    aggregate(plan.output_dir)
                    continue
            try:
                run_sequence(plan, prepared)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                _fail_remaining_sequences(plan, plan.prepared_sequences[index:], provenance, exc)
                break
            aggregate(plan.output_dir)
        return aggregate(plan.output_dir)
    finally:
        lock.unlink()
