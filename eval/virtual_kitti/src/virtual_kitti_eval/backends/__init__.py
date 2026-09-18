"""Six model keys and a read-only doctor; parent imports stay model-free."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Mapping

from .common import BackendPrediction, BackendRequest, NativeBackend, inspect_checkpoint, plain_json

MODEL_KEYS = ("vggt", "vggt_star", "streamvggt", "vggt_slam", "vggt_long", "vggt_omega")


def normalize_model_key(value):
    if not isinstance(value, str):
        raise ValueError("BACKEND_MODEL: model key must be a string")
    key = value.strip().lower().replace("-", "_")
    if key == "vggt*":
        key = "vggt_star"
    if key not in MODEL_KEYS:
        raise ValueError(f"BACKEND_MODEL: unsupported model {value!r}; expected one of {MODEL_KEYS}")
    return key


def resolve_config(model_key, config):
    key = normalize_model_key(model_key)
    if not isinstance(config, Mapping):
        raise ValueError("BACKEND_CONFIG: mapping required")
    c = plain_json(config)
    for name in ("interpreter", "project_root", "checkpoint"):
        if not isinstance(c.get(name), str) or not c[name]:
            raise ValueError(f"BACKEND_CONFIG: {name} is required")
        c[name] = str(Path(c[name]).expanduser().absolute())
    defaults = {"image_size": 512 if key == "vggt_omega" else 518, "depth_conf_thresh": 1.,
                "max_points": 0, "use_calibration": False, "loop_closure": key in ("vggt_long", "vggt_slam")}
    if key == "vggt_long":
        defaults.update(chunk_size=75, overlap=30, loop_chunk_size=20, retrieval="salad", using_sim3=True,
                        salad_batch_size=2, conf_threshold_coef=.75)
    if key == "vggt_slam":
        defaults.update(submap_size=16, max_loops=1, conf_percentile=25., lc_thres=.95)
    if key == "vggt_omega":
        defaults.update(image_mode="balanced", patch_size=16)
    for name, value in defaults.items():
        c.setdefault(name, value)
    if type(c["image_size"]) is not int or c["image_size"] != (512 if key == "vggt_omega" else 518):
        raise ValueError("BACKEND_CONFIG: image_size differs from the audited native profile")
    if type(c["max_points"]) is not int or c["max_points"] < 0:
        raise ValueError("BACKEND_CONFIG: max_points must be nonnegative integer")
    if c["use_calibration"] is not False:
        raise ValueError("BACKEND_CONFIG: these native profiles do not consume Virtual KITTI calibration")
    for name in ("depth_conf_thresh",):
        if type(c[name]) not in (int, float) or not math.isfinite(c[name]) or c[name] < 0:
            raise ValueError(f"BACKEND_CONFIG: invalid {name}")
    expected_loop = key in ("vggt_long", "vggt_slam")
    if c["loop_closure"] is not expected_loop:
        raise ValueError("BACKEND_CONFIG: loop_closure differs from the native profile")
    if key == "vggt_long":
        for name in ("dependency_path", "salad_checkpoint", "dino_checkpoint"):
            if not isinstance(c.get(name), str) or not c[name]:
                raise ValueError(f"BACKEND_CONFIG: Long requires {name}")
            c[name] = str(Path(c[name]).expanduser().absolute())
        if any(type(c[n]) is not int or c[n] != v for n, v in
               (("chunk_size", 75), ("overlap", 30), ("loop_chunk_size", 20))):
            raise ValueError("BACKEND_CONFIG: Virtual KITTI Long requires chunk 75, overlap 30, loop chunk 20")
        if c["retrieval"] != "salad" or c["using_sim3"] is not True:
            raise ValueError("BACKEND_CONFIG: Virtual KITTI Long requires SALAD and Sim3")
        if type(c["salad_batch_size"]) is not int or c["salad_batch_size"] < 1:
            raise ValueError("BACKEND_CONFIG: invalid SALAD batch size")
        if type(c["conf_threshold_coef"]) not in (float, int) or not math.isfinite(c["conf_threshold_coef"]) or c["conf_threshold_coef"] < 0:
            raise ValueError("BACKEND_CONFIG: invalid confidence coefficient")
    if key == "vggt_slam":
        if not isinstance(c.get("torch_home"), str) or not c["torch_home"]:
            raise ValueError("BACKEND_CONFIG: SLAM requires offline torch_home")
        c["torch_home"] = str(Path(c["torch_home"]).expanduser().absolute())
        if type(c["submap_size"]) is not int or c["submap_size"] < 1 or type(c["max_loops"]) is not int or c["max_loops"] != 1:
            raise ValueError("BACKEND_CONFIG: SLAM requires submap_size >= 1 and max_loops = 1")
        for name, upper in (("conf_percentile", 100), ("lc_thres", 1)):
            if type(c[name]) not in (int, float) or not math.isfinite(c[name]) or not 0 <= c[name] < upper:
                raise ValueError(f"BACKEND_CONFIG: invalid {name}")
    if key == "vggt_omega" and (type(c["patch_size"]) is not int or c["patch_size"] != 16 or c["image_mode"] not in ("balanced", "max_size")):
        raise ValueError("BACKEND_CONFIG: Omega requires patch 16 and balanced/max_size mode")
    return c


@dataclass(frozen=True)
class BackendBlocker:
    code: str
    message: str


@dataclass(frozen=True)
class BackendStatus:
    model_key: str
    ready: bool
    blockers: tuple[BackendBlocker, ...]
    diagnostics: dict


def child_environment():
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]), PYTHONDONTWRITEBYTECODE="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    return environment


def required_assets(key, c):
    root = Path(c["project_root"])
    files, directories, checkpoints = [], [root], [Path(c["checkpoint"])]
    if key in ("vggt", "vggt_star"):
        files += [root / "vggt/models/vggt.py"]
    elif key == "streamvggt":
        files += [root / "src/streamvggt/models/streamvggt.py"]
    elif key == "vggt_omega":
        files += [root / "run.py", root / "vggt_omega/models/__init__.py"]
    elif key == "vggt_long":
        files += [root / "vggt_long.py", root / "configs/base_config.yaml",
                  root / "base_models/vggt/models/vggt.py", root / "LoopModels/dinov2/hubconf.py"]
        directories += [Path(c["dependency_path"])]
        checkpoints += [Path(c["salad_checkpoint"]), Path(c["dino_checkpoint"])]
    elif key == "vggt_slam":
        files += [root / "vggt_slam/solver.py", root / "third_party/vggt/vggt/models/vggt.py",
                  root / "third_party/salad/salad/models_salad/backbones/dinov2.py",
                  Path(c["torch_home"]) / "hub/facebookresearch_dinov2_main/hubconf.py"]
        checkpoints += [Path(c["torch_home"]) / "hub/checkpoints/dino_salad.ckpt"]
    return files, directories, checkpoints


def doctor_backend(model_key: str, config: Mapping[str, object]) -> BackendStatus:
    blockers, diagnostics = [], {"allocation": "none; no model construction or CUDA initialization"}
    try:
        key = normalize_model_key(model_key)
        c = resolve_config(key, config)
    except (ValueError, TypeError) as exc:
        return BackendStatus(str(model_key), False, (BackendBlocker("BACKEND_CONFIG", str(exc)),), diagnostics)
    diagnostics["resolved_config"] = c
    files, directories, checkpoints = required_assets(key, c)
    for path in files + directories:
        if not (path.is_dir() if path in directories else path.is_file()):
            blockers.append(BackendBlocker("BACKEND_PATH_MISSING", f"missing required path: {path}"))
    diagnostics["checkpoints"] = {}
    for path in checkpoints:
        try:
            diagnostics["checkpoints"][str(path)] = inspect_checkpoint(path)
        except ValueError as exc:
            blockers.append(BackendBlocker("CHECKPOINT_INVALID", str(exc)))
    interpreter = Path(c["interpreter"])
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        blockers.append(BackendBlocker("BACKEND_INTERPRETER", f"missing executable interpreter: {interpreter}"))
    else:
        environment = child_environment()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        try:
            completed = subprocess.run([str(interpreter), "-B", "-m", "virtual_kitti_eval.backend_worker", "--doctor"],
                input=json.dumps({"model_key": key, "model_config": c}), capture_output=True, text=True,
                env=environment, timeout=90)
            probe = json.loads(completed.stdout)
            if completed.returncode != 0 or not isinstance(probe, dict) or not isinstance(probe.get("blockers"), list):
                raise ValueError(f"invalid probe result (exit {completed.returncode}): {completed.stderr[-4000:]}")
            diagnostics["probe"] = probe.get("diagnostics", {})
            blockers.extend(BackendBlocker(**record) for record in probe["blockers"])
        except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
            blockers.append(BackendBlocker("BACKEND_PROBE_FAILED", str(exc)))
    return BackendStatus(key, not blockers, tuple(blockers), diagnostics)


__all__ = ["MODEL_KEYS", "BackendRequest", "BackendPrediction", "BackendStatus", "NativeBackend",
           "normalize_model_key", "resolve_config", "doctor_backend"]
