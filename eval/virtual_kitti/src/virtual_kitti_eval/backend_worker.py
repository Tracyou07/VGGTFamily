"""Fresh-interpreter launcher and child entry point; parent imports no model package."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import zipfile
import zlib

import numpy as np

from .backends import child_environment, resolve_config
from .backends.common import BackendPrediction, BackendRequest, validate_prediction
from .io import sha256_file
from .resources import ResourceUsage, measure_inference
from .results import atomic_write_json, read_json


@dataclass(frozen=True)
class WorkerResult:
    status: str
    failure_code: str | None
    message: str
    prediction: BackendPrediction | None = None
    usage: ResourceUsage | None = None
    returncode: int | None = None
    signal: int | None = None


def _oom(message):
    return bool(re.search(r"outofmemoryerror|out of memory|cuda_error_out_of_memory|cublas_status_alloc_failed",
                          message, re.IGNORECASE))


def _failure(status, code, message, returncode=None):
    return WorkerResult(status, code, str(message), returncode=returncode,
                        signal=-returncode if returncode is not None and returncode < 0 else None)


def _tail(path, limit=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _identities_match(record, request):
    return all(record.get(name) == getattr(request, name) for name in
               ("request_id", "provenance_id", "model_key", "sequence"))


def _load_success(request, returncode):
    result_path, prediction_path = request.output_dir / "worker_result.json", request.output_dir / "prediction.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        return _failure("error", "BACKEND_OUTPUT_MISSING", "child omitted result or prediction", returncode)
    try:
        record = read_json(result_path)
        if not _identities_match(record, request):
            return _failure("error", "BACKEND_ID_MISMATCH", "request/provenance/model/sequence mismatch", returncode)
        fields = {"schema_version", "status", "failure_code", "message", "request_id", "provenance_id",
                  "model_key", "sequence", "prediction_sha256", "metadata", "resources"}
        if (set(record) != fields or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or record["status"] != "success" or record["failure_code"] is not None
            or not isinstance(record["message"], str)):
            raise ValueError("invalid success envelope")
        if sha256_file(prediction_path) != record["prediction_sha256"]:
            raise ValueError("prediction digest mismatch")
        usage = ResourceUsage(**record["resources"])
        with np.load(prediction_path, allow_pickle=False) as arrays:
            if set(arrays.files) not in ({"frame_ids", "poses_c2w"}, {"frame_ids", "poses_c2w", "world_points"}):
                raise ValueError("invalid prediction array inventory")
            ids = arrays["frame_ids"]
            if ids.ndim != 1 or ids.dtype.kind != "U":
                raise ValueError("original frame IDs must be a string vector")
            prediction = BackendPrediction(tuple(ids.tolist()), arrays["poses_c2w"],
                arrays["world_points"] if "world_points" in arrays.files else None, record["metadata"])
        validate_prediction(prediction, request.frame_ids)
        return WorkerResult("success", None, "", prediction, usage, returncode, None)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
        return _failure("error", "BACKEND_OUTPUT_INVALID", str(exc), returncode)


def launch_worker(request: BackendRequest, timeout_s: float) -> WorkerResult:
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        return _failure("error", "BACKEND_REQUEST_INVALID", "timeout must be a positive finite number")
    try:
        # Revalidate before touching output; reject every stale attempt and concurrent launch.
        request = BackendRequest.from_dict(request.to_dict())
        request.output_dir.mkdir(parents=True, exist_ok=True)
        request_path = request.output_dir / "request.json"
        if any((request.output_dir / name).exists() for name in
               ("request.json", "prediction.npz", "worker_result.json", "stdout.log", "stderr.log")):
            return _failure("error", "BACKEND_OUTPUT_EXISTS", "worker attempt directory already contains artifacts")
        fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(request.to_dict(), stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        command = [request.model_config["interpreter"], "-B", "-m", "virtual_kitti_eval.backend_worker",
                   "--request", str(request_path)]
        stdout_path, stderr_path = request.output_dir / "stdout.log", request.output_dir / "stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            child = subprocess.Popen(command, cwd=request.output_dir, env=child_environment(),
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                returncode = child.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                # Terminate the whole process group, including native helpers.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                return _failure("timeout", "BACKEND_TIMEOUT", f"worker exceeded {timeout_s} seconds", child.returncode)
        error_text = _tail(stderr_path) + "\n" + _tail(stdout_path)
        if returncode < 0:
            return _failure("error", "BACKEND_SIGNAL", error_text or "worker terminated by signal", returncode)
        if returncode != 0:
            try:
                record = read_json(request.output_dir / "worker_result.json")
                if _identities_match(record, request) and record.get("status") in ("error", "oom"):
                    if record.get("status") == "oom" or _oom(str(record.get("message", ""))):
                        return _failure("oom", "BACKEND_OOM", record.get("message", ""), returncode)
                    return _failure("error", "BACKEND_ERROR", record.get("message", ""), returncode)
            except ValueError:
                pass
            return _failure("oom" if _oom(error_text) else "error",
                            "BACKEND_OOM" if _oom(error_text) else "BACKEND_EXIT", error_text, returncode)
        return _load_success(request, returncode)
    except FileExistsError as exc:
        return _failure("error", "BACKEND_OUTPUT_EXISTS", exc)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        return _failure("error", "BACKEND_LAUNCH_FAILED", exc)


def _atomic_prediction(path, prediction):
    fd, temporary = tempfile.mkstemp(prefix=".prediction.", suffix=".tmp", dir=path.parent)
    try:
        arrays = {"frame_ids": np.asarray(prediction.frame_ids), "poses_c2w": prediction.poses_c2w}
        if prediction.world_points is not None:
            arrays["world_points"] = prediction.world_points
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def execute_request(request):
    caches = {"NUMBA_CACHE_DIR": "numba", "MPLCONFIGDIR": "matplotlib", "XDG_CACHE_HOME": "xdg"}
    previous = {key: os.environ.get(key) for key in caches}
    for key, name in caches.items():
        os.environ[key] = str(request.output_dir / "runtime_cache" / name)
    try:
        from .backends.native import create_backend
        import torch
        # Native .cuda() uses this current device; cached artifacts stay in this attempt.
        device = int(request.device.split(":")[1]) if ":" in request.device else 0
        torch.cuda.set_device(device)
        backend = create_backend(request.model_key, request.model_config, f"cuda:{device}", request.output_dir)
        prediction, usage = measure_inference(backend, request.frame_ids, request.image_paths, torch.cuda)
        return validate_prediction(prediction, request.frame_ids), usage
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _doctor_probe(payload):
    """Use disposable cache space; model/checkpoint/dependency trees stay read-only."""
    active = [True]
    with tempfile.TemporaryDirectory(prefix="virtual-kitti-doctor-") as directory:
        scratch = Path(directory).resolve()
        previous_tempdir = tempfile.tempdir
        previous = {k: os.environ.get(k) for k in ("TMPDIR", "MPLCONFIGDIR", "NUMBA_CACHE_DIR")}
        os.environ.update(TMPDIR=str(scratch), MPLCONFIGDIR=str(scratch / "matplotlib"),
                          NUMBA_CACHE_DIR=str(scratch / "numba"))
        tempfile.tempdir = str(scratch)
        try:
            return _doctor_probe_in_scratch(payload, scratch, active)
        finally:
            active[0] = False
            tempfile.tempdir = previous_tempdir
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _doctor_probe_in_scratch(payload, scratch, active):
    """Import on CPU with allocation/download guards and bytecode writes disabled."""
    import socket
    from .backends.native import install_backend_sources
    key, config = payload["model_key"], resolve_config(payload["model_key"], payload["model_config"])
    def disposable(path, dir_fd=None):
        if isinstance(path, int):
            try:
                path = os.readlink(f"/proc/self/fd/{path}").removesuffix(" (deleted)")
            except OSError:
                return False
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False
        path = Path(os.fsdecode(path))
        if not path.is_absolute() and dir_fd is not None and dir_fd != -1:
            path = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / path
        return path.resolve().is_relative_to(scratch) or path.resolve() == Path("/dev/null")

    def readonly(event, args):
        if not active[0]:
            return
        if event == "open":
            mode, flags = args[1:3]
            writing = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing and not disposable(args[0]):
                raise PermissionError(f"doctor is read-only: {args[0]}")
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"}:
            dir_fd = args[-1] if event != "os.truncate" and isinstance(args[-1], int) else None
            if not disposable(args[0], dir_fd):
                raise PermissionError(f"doctor is read-only: {event}: {args[0]}")
        if event in {"os.rename", "os.symlink", "os.link"}:
            if not all(disposable(path) for path in args[:2]):
                raise PermissionError(f"doctor is read-only: {event}")
        if event in {"subprocess.Popen", "socket.connect"}:
            raise PermissionError(f"doctor forbids subprocess/network access: {event}")
    sys.addaudithook(readonly)
    install_backend_sources(key, config)
    blockers, imported = [], {}
    def forbidden(*args, **kwargs):
        raise RuntimeError("doctor forbids CUDA initialization and network access")
    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    modules = ["torch", "numpy", "PIL", "safetensors"]
    if key in ("vggt", "vggt_star"):
        modules += ["vggt.models.vggt", "vggt.utils.load_fn", "vggt.utils.pose_enc", "vggt.utils.geometry"]
    elif key == "streamvggt":
        modules += ["streamvggt.models.streamvggt", "streamvggt.utils.load_fn", "streamvggt.utils.pose_enc", "streamvggt.utils.geometry"]
    elif key == "vggt_omega":
        modules += ["vggt_omega.models", "vggt_omega.utils.load_fn", "vggt_omega.utils.pose_enc"]
    elif key == "vggt_long":
        modules += ["yaml", "open3d", "scipy", "pytorch_lightning", "pypose", "numba", "llvmlite", "faiss",
                    "vggt_long", "base_models.vggt.models.vggt", "LoopModels.LoopModel"]
    elif key == "vggt_slam":
        modules += ["gtsam", "open3d", "scipy", "viser", "vggt_slam.solver", "salad.eval", "vggt.models.vggt"]
    for name in modules:
        try:
            module = importlib.import_module(name)
            imported[name] = str(getattr(module, "__file__", "built-in"))
            if name == "torch":
                module.cuda._lazy_init = forbidden
                module.cuda.init = forbidden
                module.hub.load = forbidden
                module.hub.download_url_to_file = forbidden
                module.hub.load_state_dict_from_url = forbidden
            if name == "gtsam":
                for attr in ("SL4", "PriorFactorSL4", "BetweenFactorSL4"):
                    if not hasattr(module, attr):
                        blockers.append({"code": "BACKEND_DEPENDENCY", "message": f"gtsam lacks {attr}; configured interpreter must support SL4"})
        except Exception as exc:
            blockers.append({"code": "BACKEND_IMPORT", "message": f"{name}: {type(exc).__name__}: {exc}"})
    return {"blockers": blockers, "diagnostics": {"python": sys.executable, "imports": imported,
            "cuda_initialization": "forbidden", "network": "forbidden", "bytecode_writes": False,
            "scratch_directory": str(scratch), "external_writes": "forbidden"}}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="virtual-kitti-backend-worker")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--doctor", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    if args.doctor:
        try:
            payload = json.load(sys.stdin)
            with redirect_stdout(sys.stderr):
                result = _doctor_probe(payload)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0
        except Exception as exc:
            print(json.dumps({"blockers": [{"code": "BACKEND_PROBE_FAILED", "message": f"{type(exc).__name__}: {exc}"}],
                              "diagnostics": {"python": sys.executable}}))
            return 0
    request = None
    try:
        request = BackendRequest.from_dict(read_json(args.request))
        if args.request.absolute() != request.output_dir / "request.json":
            raise ValueError("worker request path does not match output directory")
        for filename in ("prediction.npz", "worker_result.json"):
            if (request.output_dir / filename).exists():
                raise ValueError("worker output already exists")
        prediction, usage = execute_request(request)
        path = request.output_dir / "prediction.npz"
        _atomic_prediction(path, prediction)
        record = {"schema_version": 1, "request_id": request.request_id, "provenance_id": request.provenance_id,
                  "model_key": request.model_key, "sequence": request.sequence, "status": "success",
                  "failure_code": None, "message": "", "prediction_sha256": sha256_file(path),
                  "metadata": prediction.metadata, "resources": asdict(usage)}
        atomic_write_json(request.output_dir / "worker_result.json", record)
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        is_oom = _oom(message)
        if request is not None and not (request.output_dir / "worker_result.json").exists():
            atomic_write_json(request.output_dir / "worker_result.json", {
                "schema_version": 1, "request_id": request.request_id, "provenance_id": request.provenance_id,
                "model_key": request.model_key, "sequence": request.sequence,
                "status": "oom" if is_oom else "error", "failure_code": "BACKEND_OOM" if is_oom else "BACKEND_ERROR",
                "message": message, "prediction_sha256": None, "metadata": None, "resources": None})
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
