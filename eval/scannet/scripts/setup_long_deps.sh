#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
target="${LONG_DEPS_TARGET:-${repo_root}/.runtime/long_deps}"
mkdir -p "${target}"
"${python_bin}" -m pip install --no-deps --target "${target}" \
  'pypose==0.9.5' 'numba==0.61.2' 'llvmlite==0.44.0' 'faiss-cpu==1.8.0.post1'
PYTHONPATH="${target}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - <<'PY'
import importlib.metadata
import faiss, llvmlite, numba, pypose
for package in ("pypose", "numba", "llvmlite", "faiss-cpu"):
    print(f"{package}={importlib.metadata.version(package)}")
PY
