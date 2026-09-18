#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:-fastvggt}"
scene="${2:-scene0150_00}"
output="${3:-${repo_root}/results/smoke/${model}}"
python_bin="${SCANNET_PYTHON:-/home/ubuntu/anaconda3/envs/vggt-gx/bin/python}"
gpu_index="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICE_INDEX:-0}}"
if [[ ! "${gpu_index}" =~ ^[0-9]+$ ]]; then
  echo "CUDA_VISIBLE_DEVICES must select one physical numeric GPU, got: ${gpu_index}" >&2
  exit 2
fi
cd "${repo_root}"
"${repo_root}/scripts/preflight_h20.sh" "${gpu_index}" "${MIN_FREE_MIB:-60000}"
CUDA_VISIBLE_DEVICES="${gpu_index}" "${python_bin}" -m scannet_eval run \
  --config "${repo_root}/configs/h20.json" --model "${model}" \
  --output "${output}" --scenes "${scene}" --max-frames 8 --device cuda
