#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"
PY=${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}
"$PY" -m unittest discover -s "$ROOT/tests" -v
"$PY" -m compileall -q "$ROOT/src" "$ROOT/tests"
for f in "$ROOT"/scripts/*.sh; do bash -n "$f"; done
