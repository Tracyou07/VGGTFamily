"""Measurement-only wrapper around the unchanged ours_v5 worker."""
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import time

import torch

from experiments.ours_v5 import worker


def main():
    source = str(Path(worker.__file__).resolve())
    lines = Path(source).read_text().splitlines()
    start_line = next(i for i, line in enumerate(lines, 1) if line.strip().startswith("model=VGGT().eval()"))
    marks = {}
    original_cuda = torch.nn.Module.cuda
    def traced_cuda(model, *args, **kwargs):
        result = original_cuda(model, *args, **kwargs)
        if "start" in marks and "end" not in marks:
            torch.cuda.synchronize()
            marks["end"] = time.perf_counter()
        return result
    def trace(frame, event, arg):
        if frame.f_code.co_filename == source and event == "line" and frame.f_lineno == start_line:
            torch.cuda.synchronize()
            marks["start"] = time.perf_counter()
            sys.settrace(None)
            return None
        return trace
    original_argv = sys.argv[:]
    torch.nn.Module.cuda = traced_cuda
    sys.argv = ["ours_v5.worker"] + sys.argv[1:]
    try:
        sys.settrace(trace)
        worker.main()
    finally:
        sys.settrace(None)
        torch.nn.Module.cuda = original_cuda
        sys.argv = original_argv
    if set(marks) != {"start", "end"}:
        raise RuntimeError(f"model load measurement missing: {marks}")
    args = original_argv[1:]
    output = Path(args[args.index("--output") + 1])
    (output / "comparison_timing.json").write_text(json.dumps(
        dict(model_loading_seconds=marks["end"] - marks["start"],
             timing_notes="VGGT construction through checkpoint load and CUDA transfer; CUDA synchronized at both boundaries."), indent=2))


if __name__ == "__main__":
    main()
