"""Strict, atomic artifacts and formal 00--10 aggregation; no model imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .io import canonical_json, content_sha256
from .metrics import AteMetrics, PROTOCOL_ID, validate_frame_ids
from .provenance import validate_provenance

CANONICAL_SEQUENCES = tuple(f"{i:02d}" for i in range(11))
RESULT_FIELDS = {"schema_version", "model_key", "sequence", "status", "input_frames",
    "inference_seconds", "peak_allocated_mib", "peak_reserved_mib", "worker_exit_state",
    "provenance_id", "metrics_sha256"}
METRIC_FIELDS = {"schema_version", "protocol_id", "matched_frames", "rmse_m", "alignment",
    "frame_ids_sha256", "sequence", "model_key", "provenance_id"}
MODEL_LABELS = {"vggt": "VGGT", "vggt_star": "VGGT*", "streamvggt": "StreamVGGT",
                "vggt_slam": "VGGT-SLAM", "vggt_long": "VGGT-Long", "vggt_omega": "VGGT-Omega"}


def read_json(path: Path) -> dict:
    def object_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"DUPLICATE_JSON_KEY: {key}")
            out[key] = value
        return out
    def bad_constant(value):
        raise ValueError(f"NONFINITE_JSON: {value}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"),
                           object_pairs_hook=object_pairs, parse_constant=bad_constant)
        canonical_json(value)  # Also rejects overflowed exponent notation such as 1e999.
        if not isinstance(value, dict):
            raise ValueError("JSON_OBJECT_REQUIRED")
        return value
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"INVALID_JSON: {path}: {exc}") from exc


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    encoded = canonical_json(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metrics_record(metrics: AteMetrics, frame_ids, sequence: str, model_key: str, provenance_id: str) -> dict:
    ids = validate_frame_ids(frame_ids)
    record = {"schema_version": 1, "protocol_id": metrics.protocol_id,
        "matched_frames": metrics.matched_frames, "rmse_m": metrics.rmse_m,
        "alignment": {"scale": metrics.alignment.scale, "rotation": metrics.alignment.rotation.tolist(),
                      "translation": metrics.alignment.translation.tolist()},
        "frame_ids_sha256": content_sha256(list(ids)), "sequence": sequence, "model_key": model_key,
        "provenance_id": provenance_id}
    _validate_metrics(record)
    if metrics.matched_frames != len(ids):
        raise ValueError("FRAME_COUNT_MISMATCH")
    return record


def _number(value, label, *, nullable=False):
    if nullable and value is None:
        return
    try:
        valid = type(value) in (float, int) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"INVALID_{label}: finite nonnegative number required")


def _validate_metrics(record):
    canonical_json(record)
    if not isinstance(record, dict) or set(record) != METRIC_FIELDS:
        raise ValueError("INVALID_METRICS_SCHEMA")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1 or record["protocol_id"] != PROTOCOL_ID:
        raise ValueError("INVALID_METRICS_PROTOCOL")
    if type(record["matched_frames"]) is not int or record["matched_frames"] < 3:
        raise ValueError("INVALID_MATCHED_FRAMES")
    _number(record["rmse_m"], "RMSE")
    alignment = record["alignment"]
    if not isinstance(alignment, dict) or set(alignment) != {"scale", "rotation", "translation"}:
        raise ValueError("INVALID_ALIGNMENT")
    _number(alignment["scale"], "SCALE")
    try:
        rotation = np.asarray(alignment["rotation"], dtype=float)
        translation = np.asarray(alignment["translation"], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("INVALID_ALIGNMENT: numeric arrays required") from exc
    if (alignment["scale"] <= 0 or rotation.shape != (3, 3) or translation.shape != (3,)
        or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
        raise ValueError("INVALID_ALIGNMENT: proper positive-scale Sim3 required")


def _validate_result(result):
    canonical_json(result)
    if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
        raise ValueError("INVALID_RESULT_SCHEMA")
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("INVALID_RESULT_SCHEMA")
    if result["status"] not in ("success", "oom", "error", "timeout"):
        raise ValueError("INVALID_RESULT_STATUS")
    if type(result["input_frames"]) is not int or result["input_frames"] < 1:
        raise ValueError("INVALID_INPUT_FRAMES")
    for key in ("inference_seconds", "peak_allocated_mib", "peak_reserved_mib"):
        _number(result[key], key.upper(), nullable=result["status"] != "success")
    state = result["worker_exit_state"]
    if not isinstance(state, dict) or set(state) != {"returncode", "signal"}:
        raise ValueError("INVALID_WORKER_EXIT_STATE")
    if state["returncode"] is not None and type(state["returncode"]) is not int:
        raise ValueError("INVALID_WORKER_EXIT_STATE")
    if state["signal"] is not None and (type(state["signal"]) is not int or state["signal"] <= 0):
        raise ValueError("INVALID_WORKER_EXIT_STATE")
    if result["status"] == "success" and (state["returncode"] != 0 or state["signal"] is not None):
        raise ValueError("UNSUCCESSFUL_WORKER_EXIT")
    if result["status"] != "success" and result["metrics_sha256"] is not None:
        raise ValueError("FAILED_RESULT_MUST_NOT_REFERENCE_METRICS")


def write_result_pair(directory: Path, result: dict, metrics: dict | None) -> None:
    """Write metrics first, then its content-bound terminal result as the commit marker."""
    result = dict(result)
    if result["status"] == "success":
        if metrics is None:
            raise ValueError("MISSING_METRICS")
        _validate_metrics(metrics)
        for key in ("model_key", "sequence", "provenance_id"):
            if result[key] != metrics[key]:
                raise ValueError(f"PAIR_MISMATCH: {key}")
        if result["input_frames"] != metrics["matched_frames"]:
            raise ValueError("PAIR_FRAME_COUNT_MISMATCH")
        result["metrics_sha256"] = content_sha256(metrics)
    else:
        if metrics is not None:
            raise ValueError("FAILED_RESULT_MUST_NOT_HAVE_METRICS")
        result["metrics_sha256"] = None
    _validate_result(result)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        atomic_write_json(directory / "metrics.json", metrics)
    atomic_write_json(directory / "result.json", result)


def _result_for_sequence(directory, provenance):
    result = read_json(directory / "result.json")
    _validate_result(result)
    sequence = directory.name
    if sequence not in provenance["sequences"]:
        raise ValueError("UNEXPECTED_SEQUENCE")
    if (result["sequence"] != sequence or result["model_key"] != provenance["model_key"]
        or result["provenance_id"] != provenance["provenance_id"]
        or result["input_frames"] != len(provenance["sequences"][sequence]["frame_ids"])):
        raise ValueError("RESULT_PROVENANCE_MISMATCH")
    return result


def load_result_pair(directory: Path, provenance: dict, *, check_sources: bool = True) -> tuple[dict, dict]:
    """Resume gate: reject partial, failed, stale or mismatched artifacts.

    check_sources=False is for callers that already verified this provenance once
    in the current operation (aggregate); schema and content hashes remain checked.
    """
    validate_provenance(provenance, check_sources=check_sources)
    directory = Path(directory)
    if directory.parent.resolve() != Path(provenance["output_dir"]).resolve():
        raise ValueError("OUTPUT_PROVENANCE_MISMATCH")
    result = _result_for_sequence(directory, provenance)
    if result["status"] != "success":
        raise ValueError(f"FAILED_RESULT: {result['status']}")
    metrics = read_json(directory / "metrics.json")
    _validate_metrics(metrics)
    sequence = directory.name
    expected = provenance["sequences"][sequence]
    if (metrics["sequence"] != sequence or metrics["model_key"] != provenance["model_key"]
        or metrics["provenance_id"] != provenance["provenance_id"]
        or metrics["protocol_id"] != provenance["metric_protocol_id"]
        or metrics["matched_frames"] != result["input_frames"]
        or metrics["frame_ids_sha256"] != expected["frame_ids_sha256"]
        or content_sha256(metrics) != result["metrics_sha256"]):
        raise ValueError("METRICS_PROVENANCE_OR_PAIR_MISMATCH")
    return result, metrics


@dataclass(frozen=True)
class AggregateSummary:
    complete: bool
    model_key: str | None
    per_sequence: dict[str, float]
    avg_ate_rmse_m: float | None
    avg_star_ate_rmse_m: float | None
    failures: dict[str, dict]
    resources: dict[str, dict]


def _mean(values, sequences):
    if not all(s in values for s in sequences):
        return None
    # Divide first to avoid overflow when averaging finite nonnegative metrics.
    return math.fsum(values[s] / len(sequences) for s in sequences)


def aggregate(output_dir: Path) -> AggregateSummary:
    output_dir = Path(output_dir)
    metrics_by_sequence, resources, failures, values = {}, {}, {}, {}
    model_key = None
    try:
        provenance = read_json(output_dir / "run_manifest.json")
        validate_provenance(provenance)
        if output_dir.resolve() != Path(provenance["output_dir"]).resolve():
            raise ValueError("OUTPUT_PROVENANCE_MISMATCH")
        if provenance["metric_protocol_id"] != PROTOCOL_ID:
            raise ValueError("INVALID_METRIC_PROTOCOL")
        model_key = provenance["model_key"]
    except (ValueError, TypeError, KeyError) as exc:
        failures["_manifest"] = {"status": "invalid", "code": "INVALID_PROVENANCE", "message": str(exc)}
        provenance = None
    if provenance is not None:
        for sequence in CANONICAL_SEQUENCES:
            directory = output_dir / sequence
            try:
                result = _result_for_sequence(directory, provenance)
                if result["status"] != "success":
                    resources[sequence] = result
                    failures[sequence] = {"sequence": sequence, "status": result["status"],
                                          "worker_exit_state": result["worker_exit_state"]}
                    continue
                result, metrics = load_result_pair(directory, provenance, check_sources=False)
                resources[sequence], metrics_by_sequence[sequence] = result, metrics
                values[sequence] = metrics["rmse_m"]
            except (ValueError, KeyError, TypeError) as exc:
                failures[sequence] = {"sequence": sequence, "status": "invalid", "code": "INVALID_RESULT", "message": str(exc)}
        for directory in sorted(output_dir.iterdir()):
            if directory.is_dir() and directory.name not in CANONICAL_SEQUENCES + ("failures",):
                if (directory / "result.json").exists() or (directory / "metrics.json").exists():
                    failures[directory.name] = {"status": "invalid", "code": "UNEXPECTED_SEQUENCE"}
    summary = AggregateSummary(not failures and len(values) == 11, model_key, values,
        _mean(values, CANONICAL_SEQUENCES), _mean(values, tuple(s for s in CANONICAL_SEQUENCES if s != "01")),
        failures, resources)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = output_dir / "failures"
    failure_dir.mkdir(exist_ok=True)
    for sequence, failure in failures.items():
        atomic_write_json(failure_dir / f"{sequence}.json", failure)
    # Historical failure files are retained; summary.failures is authoritative for this aggregation.
    atomic_write_json(output_dir / "all_sequences_metrics.json", {"protocol_id": PROTOCOL_ID, "sequences": metrics_by_sequence})
    atomic_write_json(output_dir / "average_metrics.json", {"protocol_id": PROTOCOL_ID,
        "avg_ate_rmse_m": summary.avg_ate_rmse_m, "avg_star_ate_rmse_m": summary.avg_star_ate_rmse_m})
    atomic_write_json(output_dir / "summary.json", asdict(summary))
    return summary


def export_table(output_dir: Path) -> str:
    summary = aggregate(output_dir)
    display = MODEL_LABELS.get(summary.model_key, summary.model_key or "—")
    model_config = {}
    if summary.model_key:
        model_config = read_json(Path(output_dir) / "run_manifest.json")["model_config"]
    def number(value):
        return "—" if value is None else f"{value:.3f}"
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lc = model_config.get("loop_closure")
    lc = "✓" if lc is True else "✗" if lc is False else "—"
    calibrated = model_config.get("use_calibration")
    calibration = "Yes" if calibrated is True else "No Need" if calibrated is False else "—"
    row = [display, lc, calibration, "Dense" if summary.model_key in MODEL_LABELS else "—",
           number(summary.avg_ate_rmse_m), number(summary.avg_star_ate_rmse_m)]
    row.extend(number(summary.per_sequence.get(s)) for s in CANONICAL_SEQUENCES)
    row.append("complete" if summary.complete else "incomplete")
    header = "| Model | LC | Calibration | Recon. | Avg. | Avg.* | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | Status |"
    lines = [header, "|" + "|".join(["---"] * 18) + "|", "| " + " | ".join(map(cell, row)) + " |", "",
        "| Model | Sequence | Frames | Time (s) | Peak VRAM (MiB) | Status |", "|---|---|---:|---:|---:|---|"]
    for sequence in CANONICAL_SEQUENCES:
        result = summary.resources.get(sequence, {})
        status = result.get("status", summary.failures.get(sequence, {}).get("status", "unavailable"))
        cells = [display, sequence, str(result["input_frames"]) if result else "—",
                 number(result.get("inference_seconds")), number(result.get("peak_allocated_mib")), status]
        lines.append("| " + " | ".join(map(cell, cells)) + " |")
    return "\n".join(lines) + "\n"
