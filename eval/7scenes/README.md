# 7-Scenes unified evaluation

This directory contains the evaluation code used for the original deployment.
Results and run logs are not included in this source snapshot.

## Fixed paths

- Raw RGB/pose dataset: `/data/yjh/share/datasets/7scenes`
- RGB-registered evaluation depth: `/data/yjh/share/datasets/7scenes_registered_simplerecon_v1`
- VGGT checkpoint: `/data/yjh/share/pretrained/VGGT-1B/model.safetensors`
- StreamVGGT checkpoint: `/data/yjh/share/pretrained/StreamVGGT/checkpoints.pth`
- VGGT-Omega checkpoint: `/data/yjh/share/pretrained/VGGT-Omega/vggt_omega_1b_512.pt`
- Repository root: `/home/ubuntu/yjh/feedforwardreconstruct`

All methods use the FastVGGT 7-Scenes split, every-`kf` frame sampling, the
same RGB-registered ground truth, scale/shift normalization, and 0.1 m ICP.
The two reported settings are `kf=3` and `kf=10`.

## Directory layout

```text
7scenes/
├── README.md
├── adapters/       # one production adapter per model family
├── preprocessing/  # registered-depth preparation and validation
├── reference/      # FastVGGT protocol implementation
└── run_*.sh        # GPU launchers
```

## Reproduce the registered input

The prepared dataset already exists. To validate it without rewriting data:

```bash
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python \
  preprocessing/prepare_7scenes.py \
  --source-root /data/yjh/share/datasets/7scenes \
  --output-root /data/yjh/share/datasets/7scenes_registered_simplerecon_v1 \
  --verify
```

## Run the table configurations

The shell launchers take `GPU_ID KF` and automatically record an execution log
and sampled peak VRAM. Use a fresh output directory whenever invoking an
adapter directly.

```bash
# Original 24-layer VGGT (table label: VGGT)
bash run_original_vggt_job.sh GPU_ID 3
bash run_original_vggt_job.sh GPU_ID 10

# Canonical memory-optimized implementation (table label: VGGT*)
bash run_vggt_job.sh GPU_ID 3
bash run_vggt_job.sh GPU_ID 10

# StreamVGGT
bash run_streamvggt_job.sh GPU_ID 3
bash run_streamvggt_job.sh GPU_ID 10

# VGGT-SLAM with SALAD loop closure
bash run_vggtslam_job.sh GPU_ID 3
bash run_vggtslam_job.sh GPU_ID 10
```

VGGT-Long:

```bash
for kf in 3 10; do
  CUDA_VISIBLE_DEVICES=GPU_ID /home/ubuntu/anaconda3/envs/vggt-gx/bin/python \
    adapters/eval_long_7scenes.py \
    --kf "$kf" \
    --checkpoint /data/yjh/share/pretrained/VGGT-1B/model.safetensors \
    --data-root /data/yjh/share/datasets/7scenes_registered_simplerecon_v1 \
    --output-dir "results/vggtlong_kf${kf}_rerun"
done
```

VGGT-Omega:

```bash
for kf in 3 10; do
  CUDA_VISIBLE_DEVICES=GPU_ID /home/ubuntu/anaconda3/envs/vggt-gx/bin/python \
    adapters/eval_omega_7scenes.py \
    --kf "$kf" \
    --checkpoint /data/yjh/share/pretrained/VGGT-Omega/vggt_omega_1b_512.pt \
    --data-root /data/yjh/share/datasets/7scenes_registered_simplerecon_v1 \
    --output-dir "results/vggtomega_kf${kf}_rerun" \
    --device cuda:0
done
```

`VGGT kf=3` and `StreamVGGT kf=3` are expected to terminate with CUDA OOM on
an otherwise empty H20. Their OOM logs and VRAM traces are retained as the
canonical records.

## Canonical result sources

- `vggt_original_kf{3,10}`: VGGT
- `vggt_kf{3,10}`: VGGT*
- `streamvggt_kf{3,10}`: StreamVGGT
- `vggtslam_registered_v3_kf{3,10}`: VGGT-SLAM
- `vggtlong_kf{3,10}`: VGGT-Long
- `vggtomega_kf{3,10}`: VGGT-Omega

Large predicted point clouds, visualization files, smoke runs, failed protocol
variants, tests, caches, and obsolete launch logs are not required to reproduce
the table and are kept only in the timestamped archive beside this directory.
