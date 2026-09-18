#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:?usage: run_vggtslam_job.sh GPU_ID KF}
KF=${2:?usage: run_vggtslam_job.sh GPU_ID KF}
ROOT=/home/ubuntu/yjh/feedforwardreconstruct
PYTHON_BIN=/home/ubuntu/anaconda3/envs/monst3r/bin/python
DATA_ROOT=/data/yjh/share/datasets/7scenes
REGISTERED_DEPTH_ROOT=/data/yjh/share/datasets/7scenes_registered_simplerecon_v1
CHECKPOINT=/data/yjh/share/pretrained/VGGT-1B/model.safetensors
OUTPUT_DIR="$ROOT/eval/7scenes/results/vggtslam_registered_v3_kf${KF}"
LOG_DIR="$ROOT/eval/7scenes/logs"
LOG_FILE="$LOG_DIR/vggtslam_registered_v3_kf${KF}.log"
VRAM_FILE="$LOG_DIR/vram_vggtslam_registered_v3_kf${KF}.csv"

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
  --max-points 999999 2>&1 | tee "$LOG_FILE"
