#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?usage: run_model.sh MODEL GPU_ID [--resume]}
GPU=${2:?usage: run_model.sh MODEL GPU_ID [--resume]}
RESUME=${3:-}
case "$MODEL" in
  vggt|vggt_long|streamvggt|vggt_slam|vggt_omega) ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU_ID must be 0..7" >&2; exit 2; }
[[ -z "$RESUME" || "$RESUME" == --resume ]] || { echo "third argument must be --resume" >&2; exit 2; }

ROOT=/home/ubuntu/yjh/feedforwardreconstruct/eval/nrgbd
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
CONFIG="$ROOT/configs/h20.json"
EXPECTED_HOST=${H20_HOSTNAME:-VM-0-11-ubuntu}
MIN_DISK_GIB=${MIN_DISK_GIB:-20}
MIN_FREE_MIB=${MIN_FREE_MIB:-60000}
[[ "$(hostname)" == "$EXPECTED_HOST" ]] || { echo "wrong host: $(hostname), expected $EXPECTED_HOST" >&2; exit 3; }
for path in / /data; do
  free_kib=$(df -Pk "$path" | awk 'NR==2 {print $4}')
  (( free_kib >= MIN_DISK_GIB * 1024 * 1024 )) || { echo "less than ${MIN_DISK_GIB} GiB free on $path" >&2; exit 3; }
done
free_mib=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) || { echo "GPU $GPU has only ${free_mib:-unknown} MiB free" >&2; exit 3; }

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export TORCH_HOME=/data/yjh/share/pretrained/torch
"$PY" -m nrgbd_eval check --config "$CONFIG" >/dev/null
"$PY" -m nrgbd_eval doctor --config "$CONFIG" --model "$MODEL"

mkdir -p "$ROOT/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/${MODEL}_${STAMP}.log"
VRAM="$ROOT/logs/${MODEL}_${STAMP}_vram.csv"
nvidia-smi --id="$GPU" --query-gpu=timestamp,index,memory.used,memory.total --format=csv,noheader,nounits --loop-ms=500 >"$VRAM" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
/usr/bin/time -v "$PY" -m nrgbd_eval run --config "$CONFIG" --model "$MODEL" --device cuda:0 $RESUME 2>&1 | tee "$LOG"
