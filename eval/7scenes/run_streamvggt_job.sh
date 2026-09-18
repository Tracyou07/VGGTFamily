#!/usr/bin/env bash
set -u

GPU="$1"
KF="$2"
RUN_TAG="registered_v1_$(date -u +%Y%m%dT%H%M%SZ)_$$"
DATA_ROOT="${SEVENSCENES_DATA_ROOT:-/data/yjh/share/datasets/7scenes_registered_simplerecon_v1}"
ROOT="/home/ubuntu/yjh/feedforwardreconstruct"
EVAL_DIR="$ROOT/eval/7scenes"
OUT_DIR="$EVAL_DIR/results/streamvggt_kf${KF}_${RUN_TAG}"
LOG_DIR="$EVAL_DIR/logs"
LOG_FILE="$LOG_DIR/streamvggt_kf${KF}_${RUN_TAG}.log"
PID_FILE="$LOG_DIR/streamvggt_kf${KF}_${RUN_TAG}.pid"
VRAM_FILE="$LOG_DIR/vram_streamvggt_kf${KF}_${RUN_TAG}.csv"
CKPT="/data/yjh/share/pretrained/StreamVGGT/checkpoints.pth"
PYTHON_BIN="/home/ubuntu/anaconda3/envs/vggt-gx/bin/python"

mkdir -p "$OUT_DIR" "$LOG_DIR"
echo "GPU=$GPU KF=$KF CKPT=$CKPT DATA_ROOT=$DATA_ROOT OUT_DIR=$OUT_DIR" > "$LOG_FILE"
echo "timestamp,index,memory.used [MiB]" > "$VRAM_FILE"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/vggtstream/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -u "$EVAL_DIR/adapters/stream_launch_7scenes.py" \
  --weights "$CKPT" \
  --output_dir "$OUT_DIR" \
  --kf "$KF" \
  --size 518 \
  --device cuda:0 \
  --model_name StreamVGGT \
  --data_root "$DATA_ROOT" \
  >> "$LOG_FILE" 2>&1 &
JOB_PID=$!
echo "$JOB_PID" > "$PID_FILE"

while kill -0 "$JOB_PID" 2>/dev/null; do
  nvidia-smi --id="$GPU" --query-gpu=timestamp,index,memory.used --format=csv,noheader >> "$VRAM_FILE" 2>/dev/null || true
  sleep 2
done

wait "$JOB_PID"
EXIT_CODE=$?
echo "EXIT_CODE=$EXIT_CODE" >> "$LOG_FILE"
exit "$EXIT_CODE"
