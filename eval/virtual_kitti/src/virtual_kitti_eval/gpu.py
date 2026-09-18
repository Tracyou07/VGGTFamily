"""Read-only, fail-closed GPU admission before native worker launch.

The 80 GiB default is a scheduling threshold, not an inference memory estimate.
This check never sends signals and does not reserve a GPU against other launchers.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess

DEFAULT_MIN_FREE_MIB = 81920


class GPUAdmissionError(ValueError):
    def __init__(self, code, message, diagnostics):
        self.code = code
        self.diagnostics = diagnostics
        super().__init__(f"{code}: {message}")


def gpu_policy(document):
    minimum = document.get("gpu_min_free_mib", DEFAULT_MIN_FREE_MIB)
    idle = document.get("gpu_require_no_compute_processes", True)
    if type(minimum) is not int or minimum < 0 or type(idle) is not bool:
        raise ValueError("GPU_POLICY_INVALID: require nonnegative integer MiB and boolean idle policy")
    return {"gpu_min_free_mib": minimum, "gpu_require_no_compute_processes": idle}


def _query(fields):
    result = subprocess.run(["nvidia-smi", fields, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    return [tuple(v.strip() for v in row) for row in csv.reader(result.stdout.splitlines()) if row]


def _uint(value, *, positive=False):
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError("invalid integer in GPU query")
    number = int(value)
    if positive and number == 0:
        raise ValueError("expected positive integer in GPU query")
    return number


def admit_gpu(device, document=None):
    policy = gpu_policy(document or {})
    diagnostics = dict(selected_gpu_uuid=None, observed_free_mib=None,
        required_free_mib=policy["gpu_min_free_mib"], compute_processes=[],
        gpu_require_no_compute_processes=policy["gpu_require_no_compute_processes"])
    if not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise ValueError("DEVICE_INVALID: native profiles require cuda or cuda:N")
    ordinal = int(device.split(":")[1]) if ":" in device else 0
    try:
        rows = _query("--query-gpu=index,uuid,memory.total,memory.free")
        devices = []
        for row in rows:
            if len(row) != 4 or not row[1].startswith("GPU-"):
                raise ValueError("malformed GPU inventory")
            index, total, free = _uint(row[0]), _uint(row[2], positive=True), _uint(row[3])
            if free > total or any(d["index"] == index or d["uuid"] == row[1] for d in devices):
                raise ValueError("inconsistent or duplicate GPU inventory")
            devices.append(dict(index=index, uuid=row[1], total_mib=total, free_mib=free))
        if not devices:
            raise ValueError("empty GPU inventory")
        devices.sort(key=lambda d: d["index"])
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        available = devices
        if mask is not None:
            available = []
            for token in mask.split(","):
                token = token.strip()
                matches = [d for d in devices if token == str(d["index"]) or
                           (token.startswith("GPU-") and d["uuid"].startswith(token))]
                if len(matches) != 1 or matches[0] in available:
                    raise ValueError("CUDA_VISIBLE_DEVICES has unavailable, duplicate or ambiguous IDs")
                available.append(matches[0])
        if ordinal >= len(available):
            raise ValueError("logical CUDA ordinal outside visible device inventory")
        selected = available[ordinal]
        diagnostics.update(selected_gpu_uuid=selected["uuid"], physical_index=selected["index"],
                           observed_free_mib=selected["free_mib"], total_mib=selected["total_mib"])
        for row in _query("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"):
            if len(row) != 4 or row[0] not in {d["uuid"] for d in devices}:
                raise ValueError("malformed compute-process inventory")
            record = dict(gpu_uuid=row[0], pid=_uint(row[1], positive=True),
                          process_name=None if row[2] in ("", "N/A", "[Not Found]") else row[2],
                          used_mib=None if row[3] in ("N/A", "[N/A]", "[Not Supported]") else _uint(row[3]))
            if record["gpu_uuid"] == selected["uuid"]:
                diagnostics["compute_processes"].append(record)
    except (OSError, subprocess.SubprocessError, ValueError, csv.Error) as exc:
        raise GPUAdmissionError("GPU_QUERY_FAILED", str(exc), diagnostics) from exc
    if diagnostics["observed_free_mib"] < diagnostics["required_free_mib"]:
        raise GPUAdmissionError("GPU_MEMORY_INSUFFICIENT", "selected GPU has insufficient free MiB", diagnostics)
    if policy["gpu_require_no_compute_processes"] and diagnostics["compute_processes"]:
        raise GPUAdmissionError("GPU_BUSY", "selected GPU has active compute processes", diagnostics)
    return diagnostics
