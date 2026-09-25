#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/run_v6.sh GPU_ID MODE FRAMES OUTPUT [--frame-list PATH] [--scene-root PATH]" >&2
  exit 2
fi
GPU_ID="$1"; MODE="$2"; FRAMES="$3"; OUTPUT="$4"; shift 4
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
exec "$PYTHON" -B -u -m experiments.ours_v6 run --gpu "$GPU_ID" --mode "$MODE" --frames "$FRAMES" --output "$OUTPUT" "$@"
