#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "Usage: bash $0 GPU_ID KF [--sequence chess/seq-03] [--gate-only]" >&2; exit 2; fi
GPU="$1"; KF="$2"; shift 2
[[ "$GPU" =~ ^[0-9]+$ && "$KF" =~ ^(3|10)$ ]] || { echo 'GPU must be an integer; KF must be 3 or 10' >&2; exit 2; }
EVAL_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
OURS="$(cd -- "$EVAL_DIR/../../ours_v4" && pwd)"
PYTHON="/home/ubuntu/anaconda3/envs/vggt-gx/bin/python"
export PYTHONPATH="$OURS${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
# Serialize this launcher's own users; also reject all existing GPU compute jobs.
exec 9>"/tmp/ours_v4_7scenes_gpu_${GPU}.lock"
flock -n 9 || { echo "Another ours_v4 launcher holds GPU $GPU" >&2; exit 1; }
"$PYTHON" -B -c 'import json,sys; from experiments.sevenscenes.runner import gpu_preflight; print(json.dumps(gpu_preflight(sys.argv[1],sys.argv[2]),indent=2))' "$GPU" "$EVAL_DIR/results"
EXTRA=()
if [[ -n "${RESUME_DIR:-}" ]]; then
  OUT="$(realpath -e -- "$RESUME_DIR")"
  [[ "$OUT" == "$EVAL_DIR/results/ours_v4_kf${KF}_"* && -f "$OUT/run_manifest.json" ]] || { echo 'Resume path must be an existing matching ours_v4 result' >&2; exit 1; }
  EXTRA+=(--resume)
else
  RUN_TAG="$(date -u +%Y%m%dT%H%M%S%NZ)_$$"
  OUT="$EVAL_DIR/results/ours_v4_kf${KF}_${RUN_TAG}"
  mkdir -- "$OUT"
fi
echo "OUTPUT=$OUT"
if [[ ! -f "$OUT/vram.csv" ]]; then echo 'timestamp,index,uuid,memory.used,utilization.gpu' > "$OUT/vram.csv"; fi
(
  while true; do
    nvidia-smi --id="$GPU" --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$OUT/vram.csv" || true
    sleep 2
  done
) &
MONITOR=$!
trap 'kill "$MONITOR" 2>/dev/null || true; wait "$MONITOR" 2>/dev/null || true' EXIT
"$PYTHON" -B -u "$EVAL_DIR/adapters/eval_ours_v4_7scenes.py" \
  --gpu-id "$GPU" --kf "$KF" --eval-root "$EVAL_DIR" --output-dir "$OUT" \
  "${EXTRA[@]}" "$@" 2>&1 | tee -a "$OUT/raw.log"
