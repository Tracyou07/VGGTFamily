from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ("doctor", "prepare", "verify", "run", "aggregate", "export-table")

def _run_module(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "virtual_kitti_eval", *arguments],
        text=True, capture_output=True, env=environment, check=False)

def test_cli_exposes_exactly_the_required_commands():
    completed = _run_module("--help")
    assert completed.returncode == 0
    command_group = re.search(r"\{([^}]+)\}", completed.stdout)
    assert tuple(command_group.group(1).split(",")) == COMMANDS

def test_run_help_accepts_sequence_selector_and_requires_explicit_inputs():
    completed = _run_module("run", "--help")
    assert completed.returncode == 0
    assert "--sequence" in completed.stdout
    assert "--segment" not in completed.stdout
    run = _run_module("run")
    assert run.returncode == 2
    assert "--config" in run.stderr and "--model" in run.stderr and "--sequence" in run.stderr

def test_doctor_is_read_only_and_prepare_verify_operate(vkitti_131_fixture):
    f = vkitti_131_fixture
    before = sorted(p.relative_to(f.root).as_posix() for p in f.root.rglob("*"))
    result = _run_module("doctor", "--config", str(f.config_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["ready"]
    assert before == sorted(p.relative_to(f.root).as_posix() for p in f.root.rglob("*"))
    for cmd in ("prepare", "verify"):
        result = _run_module(cmd, "--config", str(f.config_path), "--sequence", "Scene01/Clone")
        assert result.returncode == 0, result.stdout + result.stderr

def test_cli_rejects_wrong_version_without_outputs(tmp_path):
    from .conftest import make_fixture
    f = make_fixture(tmp_path, version="2.0.3")
    (f.archives / "vkitti_2.0.3_rgb.tar.part").write_bytes(b"partial")
    result = _run_module("doctor", "--config", str(f.config_path))
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert {b["code"] for b in payload["sequences"][0]["blockers"]} == {
        "DATASET_VERSION_MISMATCH", "INCOMPLETE_RGB_ARCHIVE"}
    assert not f.config.prepared_root.exists()

def test_cli_aggregate_and_export(vkitti_result_dir):
    f = vkitti_result_dir
    result = _run_module("aggregate", "--output", str(f.output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["complete"]
    table = _run_module("export-table", "--output", str(f.output))
    assert table.returncode == 0, table.stdout + table.stderr
    assert table.stdout.startswith("| Model | Calibration | Scene 01 Clone |")
