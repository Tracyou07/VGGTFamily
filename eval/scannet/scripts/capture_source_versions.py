#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path


def head(path):
    run = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return run.stdout.strip() if run.returncode == 0 else None


parser = argparse.ArgumentParser(
    description="Report configured model source revisions without importing models"
)
parser.add_argument(
    "--config",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "configs" / "h20.json",
)
args = parser.parse_args()
config = json.loads(args.config.read_text())
result = {
    name: {
        "project_root": values["project_root"],
        "git_head": head(Path(values["project_root"])),
    }
    for name, values in config["models"].items()
}
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
