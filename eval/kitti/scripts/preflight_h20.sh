#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src"
export PYTHONDONTWRITEBYTECODE=1
command=("${PYTHON:-python3}" -B -m kitti_eval doctor --config "$root/configs/h20.json" "$@")
printf 'CUDA_VISIBLE_DEVICES=%q command:' "${CUDA_VISIBLE_DEVICES-<unset>}" >&2
printf ' %q' "${command[@]}" >&2
printf '\n' >&2
exec "${command[@]}"
