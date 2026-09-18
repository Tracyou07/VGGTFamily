from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ("doctor", "prepare", "verify", "run", "aggregate", "export-table")


def _run_module(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "kitti_eval", *arguments],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def test_cli_exposes_exactly_the_required_commands() -> None:
    completed = _run_module("--help")

    assert completed.returncode == 0
    command_group = re.search(r"\{([^}]+)\}", completed.stdout)
    assert command_group is not None
    assert tuple(command_group.group(1).split(",")) == COMMANDS


def test_run_help_accepts_a_sequence_selector() -> None:
    completed = _run_module("run", "--help")

    assert completed.returncode == 0
    assert "--sequence" in completed.stdout
    assert "--segment" not in completed.stdout


def test_run_requires_explicit_inputs() -> None:
    completed = _run_module("run")
    assert completed.returncode == 2
    for option in ("--config", "--model", "--sequence", "--output"):
        assert option in completed.stderr


@pytest.mark.parametrize("command", ("doctor", "prepare", "verify"))
def test_data_commands_require_an_explicit_config(command: str) -> None:
    completed = _run_module(command)
    assert completed.returncode == 2
    assert "--config" in completed.stderr


@pytest.mark.parametrize("command", ("doctor", "prepare", "verify"))
def test_data_command_help_exposes_config_and_subset(command: str) -> None:
    completed = _run_module(command, "--help")
    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--sequence" in completed.stdout


@pytest.mark.parametrize("command", ("aggregate", "export-table"))
def test_result_commands_require_output(command):
    completed = _run_module(command)
    assert completed.returncode == 2
    assert "--output" in completed.stderr
