# ours_v8: one-to-one overlap patch communication

This branch preserves the v7 model, per-window initialization and decoding,
and alignment/ownership path. It introduces only `independent` and
`overlap_correspondence`. The model wrapper defaults to `independent`; the
command-line worker requires an explicit mode. No v7 camera, patch-bank, or
register exchange is active in this path.

For each original frame present in two **adjacent** windows, each patch Query
has exactly one remote counterpart: the patch at the same grid coordinate in
the other copy of that original frame. It sees every local K/V token exactly
once and that one remote K/V token. Camera, register, and nonoverlap patch
Queries have only local K/V. Each direction reads the other's state, but
window states remain separate. More than two owners of a frame, a nonadjacent
overlap, inconsistent frame IDs, or inconsistent patch grids raise an error.

The production path applies the original global block's LayerNorm, Q/K/V
projection, Q/K normalization and RoPE to the unchanged source coordinates.
All windows complete their frame block before every window's global Q/K/V is
projected from the same pre-update states. Window order therefore cannot
expose newly updated same-layer states. The post-attention projection,
residual, LayerScale and MLP remain the original block operations.

For a corresponding Query, let `s_i` be all local scaled dot-product scores
and `s_r` its single remote score. With `m=max(max_i(s_i),s_r)`, the result is

```
(sum_i exp(s_i-m) V_i + exp(s_r-m) V_r)
    / (sum_i exp(s_i-m) + exp(s_r-m))
```

This is one joint softmax. It does not separately normalize local and remote
branches. The implementation materializes at most one local score block of
`[batch, heads, query_chunk_size, local_tokens]`, not a full-scene mask or a
per-Query copy of the local K/V. The independent dense test oracle builds an
explicit token-metadata mask and is never used by production inference.
Standard SDPA does not expose the normalization statistics needed to combine
its local output with the one remote value, so the corresponding branch uses
explicit FP32 accumulation. It converts autocast operands to BF16 first and
keeps the original V/output dtype. Local-only Queries use the native SDPA
dispatcher. There is no forced Flash scope in v8 production. The optional
`--cache-local-kv-dtype` caches the converted local K/V only while processing
one window of one global layer; its default is off. It does not reuse v7's
process-wide K conversion hook.

An opt-in `--correspondence-attention-path native_sdpa` path expresses the
same visibility with a bounded boolean mask. It copies each window's local
K/V once into a reusable `[local_tokens + query_chunk_size]` buffer, then
updates only the remote tail for each Query chunk. Native SDPA performs the
single joint softmax. The historical explicit implementation remains the
default as `--correspondence-attention-path explicit`. On the validated H20
stack, automatic dispatch uses cuDNN attention; forcing PyTorch Flash rejects
the non-null mask instead of silently falling back.

For 100 frames with windows `[0,60)`, `[30,90)`, `[60,100)` and 1,036 patch
positions per frame, the correspondence table has 124,320 **directed** Query
pairs. At chunk size 16 this is 7,772 explicit corresponding attention calls
per global layer. The bounded score memory avoids a giant dense scene mask,
but this Python/PyTorch path may be too slow at full size; no 100-frame speed,
memory, or accuracy claim has been made. A later dedicated kernel would need
to preserve this exact visibility and shared softmax denominator.

Stage-one checks (no checkpoint or real scene loaded):

```
PYTHONPATH=. python -B tests/ours_v8/test_attention.py
PYTHONPATH=. python -B -m unittest discover -s tests/ours_v8 -p test_integration.py -v
PYTHONPATH=. python -B -m unittest discover -s tests/ours_v7 -p test_alignment.py -v
CUDA_VISIBLE_DEVICES=0 V8_GPU_RESULTS=/path/to/small_gpu.json \
  PYTHONPATH=. python -B tests/ours_v8/gpu_operator.py
```

## Opt-in adjacent-window alignment objectives

The worker keeps `--alignment-mode point_legacy` as its default. Two additional
CPU objectives use the legacy Long point-map Sim(3) as their initializer:

* `point_normalized_control` changes only the point objective and optimizer.
* `point_camera_joint` adds uniformly weighted overlap camera centers and c2w
  rotations to that normalized point objective.

For the edge transform from later window B to earlier window A, the residuals
are

```
r_point  = ||s R X_B + t - X_A|| / L
r_center = ||s R C_B + t - C_A|| / L
r_rotate = angle(R_A^T R R_B)
```

`L` is frozen before optimization as the median distance of valid A points
from their coordinate-wise median. The objective is the confidence-weighted
mean Huber point loss plus unit-weighted mean Huber center and rotation losses;
all use delta 0.1. Point validity and `sqrt(conf_A * conf_B)` weights are the
legacy rules. Scale is log-parameterized, rotation uses an SO(3) exponential,
and point gradients are accumulated in bounded chunks. Nonfinite inputs,
rank-less-than-two point geometry, invalid cameras, abnormal transforms, and
optimizer non-convergence fail explicitly. No fallback is reported as a joint
alignment success. Front-window ownership and one Sim(3) per window remain
unchanged.

The frozen-output comparison is CPU only and never loads the model:

```
PYTHONPATH=. python -B -m experiments.ours_v8.compare_frozen_alignment \
  --predictions /path/to/run/windows \
  --source-manifest /path/to/run/run_manifest.json \
  --prediction-set independent \
  --mode point_camera_joint \
  --output /path/to/new_unique_result
```

The CPU oracle uses preset `atol=rtol=1e-10` for FP64 and `2e-5` for FP32.
The tiny BF16 GPU oracle uses preset `atol=rtol=0.03`, and reports both the
observed error and actual kernel names. GPU profiling is diagnostic only.

After separate authorization to run full reconstruction, use the **same**
prepared `inputs.pt` and a new output parent with sufficient free space:

```
cd /path/to/ours_v8
INPUT=/path/to/existing/fixed_100_frame/inputs.pt
OUT=/path/to/new_unique_output_parent
GPU=0
mkdir -p "$OUT"
for MODE in independent overlap_correspondence; do
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. python -B -m experiments.ours_v8.worker \
    --input "$INPUT" --output "$OUT/$MODE" --gpu "$GPU" --frames 100 \
    --window-size 60 --overlap 30 --mode "$MODE" \
    --backend-profile native_vggt --query-chunk-size 16 \
    --alignment-mode point_legacy \
    --reuse-image-encoding --cache-local-kv-dtype \
    --npz-compression-level 1 --whole-task-measurement
done
```

The worker verifies the configured checkpoint SHA256 and records source,
input, backend and correspondence metadata. It requires a clean committed
tree. The fixed 100-frame runs above are **commands only** in this phase; they
have not been executed or timed. Before executing them, recheck disk space,
GPU ownership, and whether a faster exact correspondence kernel is needed.

## Opt-in independent streaming

`--stream-independent` runs the existing independent aggregator and prediction
heads for one window at a time. Each window is transferred to CPU before the
next window starts; the existing alignment stitcher receives predictions in
the same order and retains only the preceding local prediction for the next
edge. Owned dense outputs use temporary disk-backed arrays until final NPZ
export. The temporary arrays are removed only after successful export; on
failure they remain in that run's new output directory for diagnosis.

This path requires `--mode independent --whole-task-measurement`. It rejects
`--reuse-image-encoding`, because streaming cannot preserve a cross-window GPU
feature cache. The old execution path remains the default, and
`overlap_correspondence` cannot use this option. No CUDA peak counter is reset
after model loading and no per-window `empty_cache()` is used.

To reproduce a fixed run with an existing prepared `inputs.pt`:

```
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v8
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python -B -m experiments.ours_v8.worker \
  --input /path/to/frozen/inputs.pt --output /path/to/new_unique_output \
  --gpu 4 --frames 1000 --window-size 60 --overlap 10 \
  --mode independent --backend-profile native_vggt \
  --correspondence-attention-path native_sdpa --query-chunk-size 512 \
  --alignment-mode point_camera_joint --cache-local-kv-dtype \
  --npz-compression-level 1 --whole-task-measurement \
  --stream-independent
```

The `run_manifest.json` records per-window post-forward allocated bytes and
the complete-process allocated/reserved peaks. `stream_progress.jsonl`
records each finished window; `COMPLETE.json` or `FAILED.json` marks the run.
For speed, compare `forward_seconds + overlap_stitching_seconds` with the
same fields in the historical run. `export_seconds` is separate.
