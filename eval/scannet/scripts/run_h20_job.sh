#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: run_h20_job.sh MODEL FRAMES GPU [MIN_FREE_MIB]}
FRAMES=${2:?usage: run_h20_job.sh MODEL FRAMES GPU [MIN_FREE_MIB]}
GPU=${3:?usage: run_h20_job.sh MODEL FRAMES GPU [MIN_FREE_MIB]}
MIN_FREE_MIB=${4:-90000}

case "$MODEL" in
  vggt_original|vggt_star|streamvggt|long|slam|omega) ;;
  *) echo "unsupported table model: $MODEL" >&2; exit 2 ;;
esac
case "$FRAMES" in
  100|300|500|1000) ;;
  *) echo "frames must be one of 100, 300, 500, 1000" >&2; exit 2 ;;
esac
if [[ ! "$GPU" =~ ^[0-7]$ ]]; then
  echo "GPU must be a physical index from 0 to 7" >&2
  exit 2
fi

ROOT=/home/ubuntu/yjh/feedforwardreconstruct/eval/scannet
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
if [[ "$MODEL" == "slam" ]]; then PY=/home/ubuntu/anaconda3/envs/monst3r/bin/python; fi
OUTPUT="$ROOT/results/${MODEL}_f${FRAMES}_scannet50"
LOG_DIR="$ROOT/logs"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG_FILE="$LOG_DIR/${MODEL}_f${FRAMES}_${ATTEMPT}.log"
VRAM_FILE="$LOG_DIR/${MODEL}_f${FRAMES}_${ATTEMPT}_vram.csv"

mkdir -p "$ROOT/results" "$LOG_DIR"
cd "$ROOT"
scripts/preflight_h20.sh "$GPU" "$MIN_FREE_MIB" | tee "$LOG_FILE"

echo "timestamp,index,memory.used [MiB],memory.total [MiB],utilization.gpu [%]" >"$VRAM_FILE"
nvidia-smi --id="$GPU" \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits --loop-ms=500 >>"$VRAM_FILE" &
VRAM_PID=$!
cleanup() {
  kill "$VRAM_PID" 2>/dev/null || true
  wait "$VRAM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

RESUME_ARGS=()
if [[ -f "$OUTPUT/run_manifest.json" ]]; then
  RESUME_ARGS+=(--resume)
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

set +e
"$PY" -u -m scannet_eval run \
  --config configs/h20.json \
  --model "$MODEL" \
  --max-frames "$FRAMES" \
  --output "$OUTPUT" \
  "${RESUME_ARGS[@]}" >>"$LOG_FILE" 2>&1
STATUS=$?
set -e

cleanup
trap - EXIT INT TERM
PEAK_MIB=$(awk -F, 'NR > 1 {gsub(/ /, "", $3); if ($3+0 > max) max=$3+0} END {print max+0}' "$VRAM_FILE")
echo "exit_code=$STATUS external_peak_vram_mib=$PEAK_MIB output=$OUTPUT" | tee -a "$LOG_FILE"
exit "$STATUS"
