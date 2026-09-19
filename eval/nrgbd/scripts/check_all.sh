#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ubuntu/yjh/feedforwardreconstruct/eval/nrgbd
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
export PYTHONPATH="$ROOT/src"
"$PY" -m nrgbd_eval check --config "$ROOT/configs/h20.json"
for m in vggt vggt_long streamvggt vggt_slam vggt_omega; do
 "$PY" -m nrgbd_eval doctor --config "$ROOT/configs/h20.json" --model "$m" || true
done
