#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || "$2" != "diagnose-direct" ]]; then echo "Usage: bash scripts/run_v5.sh GPU_ID diagnose-direct --output OUTPUT" >&2; exit 2; fi
GPU_ID="$1"; ACTION="$2"; shift 2
export CUDA_VISIBLE_DEVICES="$GPU_ID"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
# The launcher runs the fixed 100-frame two-mode reconstruction directly.
exec "$PYTHON" -B -u -m experiments.ours_v5 "$ACTION" --gpu "$GPU_ID" "$@"
