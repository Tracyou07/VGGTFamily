# Fixed 100-frame Long / ours_v5 comparison

This adapter runs the existing native ScanNet VGGT-Long backend and the existing
ours_v5 worker in separate processes. It does not change either model, the
checkpoint, stitching, or thresholds. The same fixed frame list and VGGT
preprocessing tensor are used; the Long worker verifies each native chunk image
tensor against the shared tensor (max absolute difference <= 1e-6).

The three runs are serial on one physical H20 GPU:

1. Native VGGT-Long, original `h20.json` configuration, 60/30 chunks.
2. ours_v5 `independent`, 60/30, batch size 2.
3. ours_v5 `camera_exchange`, 60/30, batch size 2.

Each result has an independent fresh directory. Long unaligned native chunks
are preserved inside its new output. ours keeps `windows/*/local.npz`.
`raw_window_prediction_differences.json` compares local predictions directly,
without per-window GT alignment. The common evaluator performs one global
proper Sim(3) fit per final 100-frame trajectory, then reports ATE, adjacent
relative-pose errors and exact 59→60 / 89→90 boundary errors. GT is never
passed to model inference or stitching.

Long's native code has no Git metadata. `long_manifest.json` records its
source file SHA-256, resolved native config, checkpoint hash, chunk schedule,
loop settings and input tensor checks. Long assigns overlap frames to the later
window; ours_v5 assigns them to the first window, so local prediction differences
and final trajectory differences must not be conflated.

CPU checks:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v5
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -m unittest \
  tests.ours_v5.test_compare_long -v
```

Fixed run after confirming the new commit and clean tree:

```bash
ssh h20
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v5
test "$(git rev-parse HEAD)" = "<NEW_COMMIT>" || exit 1
test -z "$(git status --porcelain)" || exit 1
export GPU_ID=4
export PYTHON=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
export RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)_long_compare"
export OUT_ROOT=/data/yjh/output/vggt/long_compare
CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -B -u -m experiments.compare_long.run_scene0150_100 \
  --gpu "$GPU_ID" --output "$OUT_ROOT/$RUN_TAG" --precision bf16 --frames 100
```

The launcher checks host/user, GPU occupancy, active compute jobs, disk,
checkpoint and clean code before creating the output. It refuses an existing
output. `comparison_summary.json`, `comparison_table.csv`,
`comparison_report.md` and `vram.csv` live at the run root. Any failed
subprocess stops the comparison; its log and `FAILED.json` remain.

Timing uses synchronized GPU boundaries for the native and ours model stages.
Reconstruction excludes model loading, GT evaluation, plotting and final
export. Long native repeated chunk preprocessing and retrieval remain within
its reconstruction time; ours preprocessing runs once per fixed input tensor.
The timing sidecars retain unclassified reconstruction time rather than
mislabeling transfers or retrieval as backbone.
