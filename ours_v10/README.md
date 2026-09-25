# ours_v6 — camera and register exchange across independent windows

v6 starts from v5 commit `bcb4885dc00d90c242accdefbe1c8fc4fa9979f0`
on branch `codex/ours-v6`. The original v5 directory and all historical
results remain unchanged. v6 reuses the same VGGT* model, checkpoint, original
heads and frozen VGGT-Long point-head Sim(3) fitter. It adds no trainable
parameter, GT-based correction, ICP, loop optimizer, CPU offload or quantization.

## What changed from v5

- Removed all public window-packing `batch_size` and `window_batch_size`
  settings. Each window is initialized, frame-attended and decoded separately.
- All windows remain at the same network layer; their communication tokens
  exchange through a same-layer snapshot before moving to the next layer.
- Added `camera_register_exchange`. `independent` and `camera_exchange`
  remain as equal-schedule controls.
- Kept the tested v5 Long fitter and first-window ownership. Window B is
  transformed into preceding window A's coordinates; A never moves to B.
- Recorded packed confidence masks, local predictions, adjacent/cumulative
  Sim(3), ownership, all adjacent errors, actual boundary errors, internal
  errors, common pairs, stage timing and memory estimates.

At each original global block:

| Query | Own window all tokens | Remote camera | Remote register | Remote patch |
| --- | --- | --- | --- | --- |
| Camera, v6 main mode | yes | yes | yes | no |
| Register, v6 main mode | yes | yes | yes | no |
| Patch, v6 main mode | yes | no | no | no |

`camera_exchange` gives only camera queries remote camera keys. `independent`
does not add remote keys. All three use identical preprocessing, windows,
first-frame reference *per window*, heads, Sim(3) and evaluation.

`vggt/v6/attention.py` projects only the permitted remote camera/register
rows and runs attention against allowed submatrices. It does not allocate a
scene-sized attention mask or score matrix. Every window has its own token
instance, including duplicate overlap frames. All remote K/V rows are
captured before any global-block writeback. The following original frame
block carries special-token information into local patch tokens.

The full current window states and required DPT head caches stay on GPU.
The manifest records estimated live state, special bank, temporary QKV,
concatenated K/V and head-cache bytes; CUDA allocated/reserved peaks are
measured separately. No 1000-frame memory claim is made.

## Stitching and evaluation

For adjacent windows A then B, the unchanged
`vggt/v5/alignment.py` estimator returns `S_B_to_A` from equal-frame,
equal-pixel point-head correspondences. v6 records its exact confidence
selection mask as a packed array beside the edge JSON. Cumulative transforms
are `S_B_to_global = S_A_to_global ∘ S_B_to_A`; W0 is identity.
C2W centers, rotations, points and depth are transformed consistently;
intrinsics are unchanged. Overlap frames align windows but the first
window to own each frame supplies the final pose/depth/intrinsics/confidence/
point-head output. `global_predictions.npz` stores the one-owner-per-frame
pose, depth, intrinsics, confidence and point map together.

The current 100-frame schedule is `[0,60)`, `[30,90)`, `[60,100)`.
Ownership is W0: 0–59, W1: 60–89, W2: 90–99. The evaluator fits one
whole-trajectory Sim(3) to GT after reconstruction. Its
`evaluation_summary.json` distinguishes actual ownership boundaries,
within-window edges, and common requested 59→60 / 89→90 pairs. VGGT-Long
uses later-window ownership; its 89→90 pair is not a true ownership
boundary, so do not compare those boundary aggregates as identical events.

## CPU development checks

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v6
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -m unittest discover -s tests/ours_v6 -v
```

## Future H20 GPU gates — not executed in this development round

Before any GPU run, inspect host identity, GPU occupancy, active jobs, disk,
code commit, clean worktree and Python environment. Use an idle physical GPU.
The launcher checks these again and refuses existing output directories.
Run each mode in a fresh run ID. Set `GPU_ID` to an idle card and choose
a fresh `RUN_TAG`; `PYTHON` is optional.

```bash
ssh h20
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v6
test "$(git branch --show-current)" = codex/ours-v6 || exit 1
test -z "$(git status --porcelain)" || exit 1
nvidia-smi
df -h /data
export GPU_ID=4
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)_v6"
export OUT_ROOT=/data/yjh/output/vggt/ours_v6
export PYTHON=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
```

First, a 25-frame single-window equality check across the three modes:

```bash
for MODE in independent camera_exchange camera_register_exchange; do
  bash scripts/run_v6.sh "$GPU_ID" "$MODE" 25 "$OUT_ROOT/${RUN_TAG}_single_${MODE}" || exit 1
done
"$PYTHON" -B -m experiments.ours_v6.check_runs --kind single \
  --independent "$OUT_ROOT/${RUN_TAG}_single_independent" \
  --camera-exchange "$OUT_ROOT/${RUN_TAG}_single_camera_exchange" \
  --camera-register-exchange "$OUT_ROOT/${RUN_TAG}_single_camera_register_exchange"
```

Then a minimal two-window 61-frame check, with a 31-frame tail:

```bash
for MODE in independent camera_exchange camera_register_exchange; do
  bash scripts/run_v6.sh "$GPU_ID" "$MODE" 61 "$OUT_ROOT/${RUN_TAG}_multi_${MODE}" || exit 1
done
"$PYTHON" -B -m experiments.ours_v6.check_runs --kind multi \
  --independent "$OUT_ROOT/${RUN_TAG}_multi_independent" \
  --camera-exchange "$OUT_ROOT/${RUN_TAG}_multi_camera_exchange" \
  --camera-register-exchange "$OUT_ROOT/${RUN_TAG}_multi_camera_register_exchange"
```

Only after those gates, the fixed 100-frame comparison:

```bash
for MODE in independent camera_exchange camera_register_exchange; do
  bash scripts/run_v6.sh "$GPU_ID" "$MODE" 100 "$OUT_ROOT/${RUN_TAG}_f100_${MODE}" || exit 1
done
"$PYTHON" -B -m experiments.ours_v6.check_runs --kind multi \
  --independent "$OUT_ROOT/${RUN_TAG}_f100_independent" \
  --camera-exchange "$OUT_ROOT/${RUN_TAG}_f100_camera_exchange" \
  --camera-register-exchange "$OUT_ROOT/${RUN_TAG}_f100_camera_register_exchange"
```

The default frame list is the verified v5
`configs/scene0150_00_frames100.json`; the launcher cross-checks its SHA-256,
frame order, checkpoint SHA-256 and preprocessing against the archived v5
manifest before running. For other sequence lengths or scenes, pass
`--scene-root` and `--frame-list`; no frame is padded or repeated.
`--batch-size` and `--window-batch-size` are rejected.

The read-only gate checker also prints per-window numeric differences against
`independent`; BF16 call-shape changes are not labeled as communication gains.

This round validates CPU behavior only. BF16 GPU equivalence, multi-window
numerics, speed, peak VRAM and 100-frame ATE remain unverified.


## 61-frame communication diagnosis

The optional read-only probe uses the same fixed input tensor as a prior v6 run. It runs one real global block with normal and zeroed remote K/V banks, then records sampled per-head attention weights at every global layer for all three modes. It does not run prediction heads, stitching, or GT evaluation. Diagnostic attention weights are recomputed in FP32 from the actual Q/K, because SDPA does not expose its internal weight matrix; production outputs remain BF16 SDPA. The four evenly spaced frames sampled per window and their query types are recorded in JSON. The probe writes only small JSON artifacts under a fresh output directory.

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v6
export CUDA_VISIBLE_DEVICES=4
PYTHON=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
INPUT=/data/yjh/output/vggt/ours_v6/20260921T081824Z_v6_f100_direct_independent/inputs.pt
OUTPUT=/data/yjh/output/vggt/ours_v6/$(date -u +%Y%m%dT%H%M%SZ)_communication_probe
"$PYTHON" -B -m experiments.ours_v6.communication_probe --gpu 4 --input "$INPUT" --output "$OUTPUT"
```
