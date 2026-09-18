import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kitti_eval.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def standalone_config(tmp_path):
    root = tmp_path / "standalone"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    for name in ("h20.json", "sequences.txt"):
        (config_dir / name).write_bytes((ROOT / "configs" / name).read_bytes())
    return load_config(config_dir / "h20.json")


def test_long_dependency_resolves_to_model_owned_runtime(standalone_config):
    expected = Path("/home/ubuntu/yjh/feedforwardreconstruct/vggtlong/.runtime/long_deps")
    assert Path(standalone_config.models["vggt_long"]["dependency_path"]).resolve() == expected


def test_doctor_reports_the_missing_local_dependency(tmp_path, standalone_config):
    from kitti_eval.backends import doctor_backend
    model = dict(standalone_config.models["vggt_long"])
    for key in ("interpreter", "project_root", "checkpoint", "salad_checkpoint", "dino_checkpoint"):
        model[key] = str(tmp_path / key)
    expected = tmp_path / "missing-long-deps"
    model["dependency_path"] = str(expected)
    status = doctor_backend("vggt_long", model)
    assert not status.ready
    assert any(blocker.code == "BACKEND_PATH_MISSING"
               and Path(blocker.message.removeprefix("missing required path: ")).resolve() == expected
               for blocker in status.blockers)


def test_offline_setup_targets_only_the_local_ignored_cache(tmp_path):
    script = ROOT / "scripts" / "setup_long_deps.sh"
    assert script.is_file()
    interpreter = tmp_path / "capture-python"
    capture = tmp_path / "arguments.json"
    interpreter.write_text("#!" + sys.executable + "\nimport json, os, sys\n"
                           "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n")
    interpreter.chmod(0o755)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    completed = subprocess.run(["bash", str(script), str(wheelhouse)],
        env={**os.environ, "PYTHON": str(interpreter), "CAPTURE": str(capture)},
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    args = json.loads(capture.read_text())
    assert args[:3] == ["-m", "pip", "install"]
    assert "--no-index" in args and "--no-deps" in args
    assert args[args.index("--find-links") + 1] == str(wheelhouse)
    assert args[args.index("--target") + 1] == "/home/ubuntu/yjh/feedforwardreconstruct/vggtlong/.runtime/long_deps"
    assert args[-4:] == ["faiss-cpu==1.8.0.post1", "llvmlite==0.44.0",
                        "numba==0.61.2", "pypose==0.9.5"]
