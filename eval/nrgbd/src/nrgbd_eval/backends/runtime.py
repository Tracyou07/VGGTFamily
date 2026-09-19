from pathlib import Path
import json
import os
import subprocess
import time
import numpy as np
from ..contracts import ScenePrediction
from .common import file_sha256, source_sha256


def doctor_backend(name, config):
    root = Path(config["source_root"])
    checkpoint = Path(config["checkpoint"])
    python = Path(config["python"])
    required = {
        relative: (root / relative).is_file()
        for relative in config.get("required_files", [])
    }
    helper = config.get("settings", {}).get("adapter_helper")
    helper_ok = not helper or Path(helper).is_file()
    required_paths = {
        path: Path(path).exists() for path in config.get("required_paths", [])
    }
    dependency_ok = False
    dependency_error = None
    if python.exists():
        probe = (
            "import importlib.util,sys;"
            "missing=[x for x in sys.argv[1:] if importlib.util.find_spec(x) is None];"
            "raise SystemExit('missing:'+','.join(missing) if missing else 0)"
        )
        env = os.environ.copy()
        extra = [str(root), str(root / "src"), str(root / "base_models")]
        env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")])
        result = subprocess.run(
            [str(python), "-c", probe, *config.get("dependencies", [])],
            text=True,
            capture_output=True,
            env=env,
        )
        dependency_ok = result.returncode == 0
        dependency_error = result.stderr.strip() or result.stdout.strip() or None
    checks = {
        "source_root": root.is_dir(),
        "checkpoint": checkpoint.is_file(),
        "python": python.exists(),
        "worker": bool(config.get("worker_module", "nrgbd_eval.backend_worker")),
        "required_files": all(required.values()),
        "adapter_helper": helper_ok,
        "required_paths": all(required_paths.values()),
        "dependencies": dependency_ok,
    }
    return {
        "model": name,
        "ready": all(checks.values()),
        "checks": checks,
        "required_files": required,
        "required_paths": required_paths,
        "dependency_error": dependency_error,
        "source_root": str(root),
        "checkpoint": str(checkpoint),
        "python": str(python),
    }


class RuntimeBackend:
    def __init__(self, name, config, device):
        self.name = name
        self.config = dict(config)
        self.device = str(device)
        self._source_sha256 = None
        self._checkpoint_sha256 = None

    def provenance(self):
        if self._source_sha256 is None:
            self._source_sha256 = source_sha256(self.config["source_root"])
        if self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = file_sha256(self.config["checkpoint"])
        return {
            "model": self.name,
            "source_sha256": self._source_sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
            "implementation_sha256": source_sha256(Path(__file__).parents[1]),
        }

    def predict(self, scene, work_dir):
        report = doctor_backend(self.name, self.config)
        if not report["ready"]:
            raise RuntimeError(f"backend doctor failed: {report}")
        work = Path(work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        req = {
            "schema_version": 1,
            "model": self.name,
            "scene_id": scene.scene_id,
            "frame_ids": list(scene.frame_ids),
            "rgb_paths": [str(p) for p in scene.rgb_paths],
            "device": self.device,
            "checkpoint": str(Path(self.config["checkpoint"]).resolve()),
            "source_root": str(Path(self.config["source_root"]).resolve()),
            "output": str(work / "prediction.npz"),
            "settings": self.config.get("settings", {}),
        }
        request = work / "request.json"
        request.write_text(json.dumps(req, indent=2, sort_keys=True) + "\n")
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(Path(__file__).parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        )
        start = time.perf_counter()
        p = subprocess.run(
            [
                self.config["python"],
                "-m",
                self.config.get("worker_module", "nrgbd_eval.backend_worker"),
                "--request",
                str(request),
            ],
            text=True,
            capture_output=True,
            env=env,
        )
        (work / "worker.stdout.log").write_text(p.stdout)
        (work / "worker.stderr.log").write_text(p.stderr)
        if p.returncode:
            raise RuntimeError(
                f"{self.name} worker failed ({p.returncode}); see {work}"
            )
        with np.load(req["output"], allow_pickle=False) as z:
            points = z["world_points"]
            masks = z["valid_masks"]
            aggregate = z["aggregate_points"] if "aggregate_points" in z.files else None
            infer = float(z["inference_seconds"])
            allocated = int(z["peak_allocated_bytes"])
            reserved = int(z["peak_reserved_bytes"])
        if self._source_sha256 is None:
            self._source_sha256 = source_sha256(req["source_root"])
        if self._checkpoint_sha256 is None:
            self._checkpoint_sha256 = file_sha256(req["checkpoint"])
        metadata = {
            "model": self.name,
            "source_sha256": self._source_sha256,
            "checkpoint_sha256": self._checkpoint_sha256,
            "worker_module": self.config.get(
                "worker_module", "nrgbd_eval.backend_worker"
            ),
            "worker_wall_seconds": time.perf_counter() - start,
        }
        return ScenePrediction(
            scene.scene_id,
            scene.frame_ids,
            points,
            masks,
            infer,
            metadata["worker_wall_seconds"],
            allocated,
            reserved,
            metadata,
            aggregate,
        )
