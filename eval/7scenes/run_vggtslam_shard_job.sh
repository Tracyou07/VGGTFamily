#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?usage: run_vggtslam_shard_job.sh GPU_ID KF SHARD SEQUENCE...}
KF=${2:?usage: run_vggtslam_shard_job.sh GPU_ID KF SHARD SEQUENCE...}
SHARD=${3:?usage: run_vggtslam_shard_job.sh GPU_ID KF SHARD SEQUENCE...}
shift 3
if (($# == 0)); then
  echo "At least one scene/sequence is required" >&2
  exit 2
fi
if [[ ! "$SHARD" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid shard name: $SHARD" >&2
  exit 2
fi

ROOT=/home/ubuntu/yjh/feedforwardreconstruct
PYTHON_BIN=/home/ubuntu/anaconda3/envs/monst3r/bin/python
DATA_ROOT=/data/yjh/share/datasets/7scenes
REGISTERED_DEPTH_ROOT=/data/yjh/share/datasets/7scenes_registered_simplerecon_v1
CHECKPOINT=/data/yjh/share/pretrained/VGGT-1B/model.safetensors
OUTPUT_DIR="$ROOT/eval/7scenes/results/vggtslam_registered_v3_kf${KF}_${SHARD}"
LOG_DIR="$ROOT/eval/7scenes/logs"
LOG_FILE="$LOG_DIR/vggtslam_registered_v3_kf${KF}_${SHARD}.log"
VRAM_FILE="$LOG_DIR/vram_vggtslam_registered_v3_kf${KF}_${SHARD}.csv"

mkdir -p "$LOG_DIR"

nvidia-smi --id="$GPU_ID" \
  --query-gpu=timestamp,memory.used,memory.total \
  --format=csv,noheader,nounits --loop-ms=500 >"$VRAM_FILE" &
VRAM_PID=$!
trap 'kill "$VRAM_PID" 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TORCH_HOME=/data/yjh/share/pretrained/torch
export PYTHONPATH="$ROOT/vggtslam/third_party/vggt:$ROOT/vggtslam/third_party/salad:$ROOT/vggtslam:$ROOT:${PYTHONPATH:-}"

SEQUENCE_ARGS=()
for sequence in "$@"; do
  SEQUENCE_ARGS+=(--sequence "$sequence")
done

"$PYTHON_BIN" -u "$ROOT/eval/7scenes/adapters/eval_vggtslam_7scenes.py" \
  --kf "$KF" \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --registered-depth-root "$REGISTERED_DEPTH_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --submap-size 16 \
  --max-loops 1 \
  --lc-thres 0.95 \
  --max-points 999999 \
  "${SEQUENCE_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
