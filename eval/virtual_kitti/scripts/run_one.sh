#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then
  printf 'Usage: %s MODEL SEQUENCE OUTPUT [--config PATH] [--device cuda:N] [--timeout SECONDS] [--resume]\n' "$0" >&2
  exit 2
fi
model="$1"; sequence="$2"; output="$3"; shift 3
if [[ ! "$sequence" =~ ^Scene(01|02|06|18|20)/(Clone|Fog|Morning|Overcast|Rain|Sunset)$ ]]; then
  printf 'SEQUENCE must be one explicit SceneXX/Condition ID\n' >&2
  exit 2
fi
for argument in "$@"; do
  if [[ "$argument" == --sequence || "$argument" == --sequence=* ]]; then
    printf 'run_one.sh accepts exactly one positional sequence\n' >&2
    exit 2
  fi
done
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src"
export PYTHONDONTWRITEBYTECODE=1
command=("${PYTHON:-python3}" -B -m virtual_kitti_eval run --config "$root/configs/h20.json"
         --model "$model" --sequence "$sequence" --output "$output" "$@")
printf 'CUDA_VISIBLE_DEVICES=%q command:' "${CUDA_VISIBLE_DEVICES-<unset>}" >&2
printf ' %q' "${command[@]}" >&2
printf '\n' >&2
exec "${command[@]}"
