#!/usr/bin/env bash
set -euo pipefail
gpu_index="${1:-${CUDA_DEVICE_INDEX:-${CUDA_VISIBLE_DEVICES:-0}}}"
min_free_mib="${2:-${MIN_FREE_MIB:-60000}}"
if [[ ! "${gpu_index}" =~ ^[0-9]+$ ]]; then
  echo "GPU index must be one physical numeric index, got: ${gpu_index}" >&2
  exit 2
fi
if [[ "$(hostname)" != "VM-0-11-ubuntu" || "$(id -un)" != "ubuntu" ]]; then
  echo "unexpected host identity: $(id -un)@$(hostname)" >&2
  exit 2
fi
df -h / /data 2>/dev/null || df -h /
nvidia-smi --id="${gpu_index}" --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo "active compute processes on physical GPU ${gpu_index}:"
nvidia-smi --id="${gpu_index}" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
free_mib="$(nvidia-smi --id="${gpu_index}" --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR == 1 { print $1 }')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || [[ "${free_mib}" -lt "${min_free_mib}" ]]; then
  echo "physical GPU ${gpu_index} has ${free_mib:-unknown} MiB free; ${min_free_mib} MiB required" >&2
  exit 1
fi
echo "preflight passed: physical GPU ${gpu_index} has ${free_mib} MiB free"
