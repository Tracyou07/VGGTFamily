#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )) || [[ "$1" == --help ]]; then
  printf 'Usage: %s OFFLINE_WHEELHOUSE\nInstall the four pinned Long dependencies into the configured model-owned runtime.\nSet PYTHON to the configured native model interpreter before running.\n' "$0"
  [[ "${1-}" == --help ]] && exit 0
  exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
wheelhouse="$(cd -- "$1" && pwd)"
target="$(python3 - "$root/configs/h20.json" <<'PY'
import json
from pathlib import Path
import sys
config = Path(sys.argv[1]).resolve()
value = json.loads(config.read_text())["models"]["vggt_long"]["dependency_path"]
print((config.parent / value).absolute())
PY
)"
exec "${PYTHON:?Set PYTHON to the configured native model interpreter}" -m pip install \
  --no-index --find-links "$wheelhouse" --no-deps --target "$target" \
  'faiss-cpu==1.8.0.post1' 'llvmlite==0.44.0' 'numba==0.61.2' 'pypose==0.9.5'
