# ours_v7: camera + deterministic patch exchange

This branch starts from v6 commit `b10372dbbae78c4baafcf5d9a95e8e2393b88b22`. The old v6 package, its results, checkpoint, and datasets are unchanged. V7 reuses the original VGGT* frame blocks, global-block Q/K/V weights, normalization, RoPE, output projection, residuals, MLP, CameraHead, depth head, point head, and v6's VGGT-Long point-head overlap Sim(3) stitching. No new trainable parameters are introduced.

Each overlap window has an independent token state and its own first-frame reference token. For every original global block, all frame blocks complete first. Read-only K/V banks are then built from the same pre-global-block state of each window. The three attention modes are:

| Mode | Cross-window queries | Remote K/V |
| --- | --- | --- |
| `independent` | none | none |
| `camera_exchange` | camera | camera |
| `camera_patch_exchange` | camera and selected patch | camera and selected patch |

Every query still sees every token in its own window. Unselected patch and register queries only see their own window in the same global block. Allowed local and remote K/V enter **one softmax**. The implementation issues grouped SDPA calls without a full-scene square attention matrix or dense mask. The original frame attention stays local to each frame. Information can reach ordinary patches indirectly in later blocks.

Patch selection is `farthest_grid_v1`: normalized 2D patch-center coordinates, center-nearest initial point, greedy farthest-point sampling, lowest row-major index for a tie. The selected count is exactly `ceil(ratio × H_patch × W_patch)`, clamped to the grid size. The same sorted indices are used for every frame, overlap copy, and aggregator layer. The ratio can be 0 (exact camera-only behavior) or 1. Config and run manifest record grid, algorithm, selected positions, requested and actual ratios. Source RoPE positions are indexed, not renumbered.

Window schedule, local heads, weighted point-head Sim(3), transform direction B-local → A-local, first-window frame ownership, and GT evaluation are inherited unchanged. With 100 frames and 60/30 windows, ownership is W0 [0,60), W1 [60,90), W2 [90,100). There is no `window_batch_size` parameter. V7 currently keeps all window states and required head caches on the compute device; it does not yet offload them to CPU. Runtime and peak VRAM of cross-window patch exchange, especially for long sequences, have **not** been measured.

## CPU tests

Run on H20 without using the GPU:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v7
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -m unittest discover -s tests/ours_v7 -v
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -m unittest discover -s tests/ours_v6 -v
```

The v7 tests use small fixtures, synthetic point maps, and no checkpoint. They check the dense-mask reference, direct gradient dependencies, no full-scene scores, ratio 0/1, sampling count and coverage, layer reuse, order independence, separate overlap states, original single-window heads, BF16 CPU where supported, and B→A/first-window stitching including a short tail. Passing them does not establish real-image GPU BF16 equivalence or reconstruction quality.

## Real-data commands for a later round (not run in this development task)

The launcher checks the host/GPU/disk, clean committed source, checkpoint identity, and fixed input provenance. It creates fresh directories under `/data/yjh/output/vggt/ours_v7`. Set an actually free GPU before running.

```bash
ssh h20
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v7
export GPU_ID=4
export PYTHON=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)_v7_f100"
export OUT_ROOT=/data/yjh/output/vggt/ours_v7

bash scripts/run_v7.sh "$GPU_ID" independent 100 "$OUT_ROOT/${RUN_TAG}_independent"
bash scripts/run_v7.sh "$GPU_ID" camera_exchange 100 "$OUT_ROOT/${RUN_TAG}_camera_exchange"
bash scripts/run_v7.sh "$GPU_ID" camera_patch_exchange 100 "$OUT_ROOT/${RUN_TAG}_camera_patch_exchange" --patch-exchange-ratio 0.10
```

Each mode runs separately. Use `--window-size`, `--overlap`, and `--frame-list` only when intentionally changing the experiment. Inputs and settings are in `configs/v7_validation.json`. Results include `config.json`, `run_manifest.json`, window predictions, alignment diagnostics, final trajectory, evaluation, and `COMPLETE.json` or `FAILED.json`.
