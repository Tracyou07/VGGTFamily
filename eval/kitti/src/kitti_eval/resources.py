"""One synchronized inference boundary; importing this module never imports torch."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import subprocess
import time

from .io import canonical_json


@dataclass(frozen=True)
class ResourceUsage:
    inference_seconds: float
    peak_allocated_mib: float
    peak_reserved_mib: float
    nvidia_smi_before: dict
    nvidia_smi_after: dict

    def __post_init__(self):
        for name in ("inference_seconds", "peak_allocated_mib", "peak_reserved_mib"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"RESOURCE_INVALID: {name}")
        for name in ("nvidia_smi_before", "nvidia_smi_after"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"RESOURCE_INVALID: {name}")
            canonical_json(getattr(self, name))


def _smi(label):
    record = {"label": label, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        record.update(returncode=completed.returncode, stdout=completed.stdout.strip(),
                      stderr=completed.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        record.update(returncode=None, error=str(exc))
    return record


def measure_inference(backend, frame_ids, image_paths, cuda, clock=time.perf_counter):
    """Loading and external snapshots are excluded; all native inference work is included."""
    backend.load()
    cuda.reset_peak_memory_stats()
    before = _smi("before")
    cuda.synchronize()
    started = clock()
    prediction = backend.infer(frame_ids, image_paths)
    cuda.synchronize()
    elapsed = clock() - started
    allocated = cuda.max_memory_allocated() / 2**20
    reserved = cuda.max_memory_reserved() / 2**20
    after = _smi("after")
    return prediction, ResourceUsage(elapsed, allocated, reserved, before, after)
