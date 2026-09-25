# ours_v4: 7Scenes unified protocol

Versioned implementation: `ours_v4/experiments/sevenscenes`; deployed adapter and
launcher copies: `eval/7scenes`. Starting commit: a3b13e561de97b0699e42f1cd29fe0918b5483ce.
No original model, ScanNet code, old adapter or historical result is overwritten.

## Formal commands (18 sequences each)

Replace GPU_ID with an idle physical GPU index. Run serially on that GPU.

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes
bash run_ours_v4_job.sh GPU_ID 3
bash run_ours_v4_job.sh GPU_ID 10
```

The launcher checks H20 identity, active compute processes, GPU memory/utilization
and at least 10 GiB free on the output filesystem. A flock prevents simultaneous
ours_v4 launches on the same GPU. It generates a fresh timestamp/PID directory,
captures raw.log and samples the selected GPU UUID/VRAM every two seconds.
It never kills another job. Dataset registration and all registered-depth checksums
are validated before loading the model. No new data/checkpoint download.

## Development checks (not full benchmark)

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v4
PYTHON=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
OMP_NUM_THREADS=1 "$PYTHON" -B -m unittest discover -s tests/ours_v4 -v
OMP_NUM_THREADS=1 "$PYTHON" -B -m unittest discover -s tests -v

cd /home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes
# 55 sampled frames -> 30,30,15. Two separate inference/stitch paths, batch=1/2.
bash run_ours_v4_job.sh GPU_ID 10 --sequence chess/seq-03 --gate-only
# Run only after the above gate passed. Entire sampled sequence, no frame cap.
bash run_ours_v4_job.sh GPU_ID 10 --sequence chess/seq-03
```

Predeclared BF16 gate tolerance: elementwise atol=0.02, rtol=0.02; raw-coordinate
camera-center distance <=0.01; rotation <=0.5 degrees. Both local predictions and
the stitched result must pass, without any GT alignment. These are the established
v4 budgets, not fitted to this sequence. NaN/Inf is rejected.
Gate success writes GATE_COMPLETE.json, not the formal completion marker.

## Resume

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes
RESUME_DIR="$PWD/results/ours_v4_kf3_EXISTING_TAG" bash run_ours_v4_job.sh GPU_ID 3
```

Use the original kf and sequence selection, identical runtime source/dependencies
and input provenance. Resume compares the manifest fingerprint; changed settings
are rejected, without quarantining other results. Verified completed attempts are
skipped. Missing/corrupted/failed attempts are retained and retried in a new
attempt_NNNN directory. COMPLETE contains hashes of metrics, diagnostics, trajectory,
window cameras and all alignment artifacts. sequences.jsonl is regenerated from
verified rows, preventing duplicate entries. Summary always lists valid_sequences
and missing sequences. Only 18/18 produces top-level COMPLETE.json; one-sequence
smoke produces SMOKE_COMPLETE.json with formal_complete=false. Old failures remain
in attempt folders; RECOVERED.json records successful recovery. A stale top-level
completion marker is archived before an incomplete resume.

## Fixed inference and evaluation

- Original frozen VGGT* and original camera/depth/point heads, BF16 autocast;
  independent 30-frame windows, overlap 10, batch 2 for equal lengths, no padding
  or shared contextual features. First-window ownership, independent reference frame
  per window. Checkpoint SHA256 is checked before model load.
- Exact v4 RANSAC/Umeyama: threshold .05, iterations 256, min_inliers 100,
  min_ratio .25, max_rmse .05, seed 17, pixel_stride 8, max_points 20000,
  confidence_quantile .5; remaining original defaults are in the manifest.
- Same original Stitcher class is loaded in a private module. Only diagnostic
  writers are deferred until compute timing stops; no geometry is reimplemented.
  A regression test compares it with the original Stitcher.
- After applying each window's Sim3 consistently to depths and camera centers,
  preserving proper camera rotations and unchanged intrinsics, depth unprojection
  produces the global point map. GT is loaded only after reconstruction finishes.
- The existing SevenScenes loader supplies the test split, kf sampling and registered
  GT at 518x392. The metric function is copied verbatim from the existing Long adapter,
  retaining Regr3D_t_ScaleShiftInv(L21,norm_mode=False,gt_scale=True), central 224x224,
  deterministic 999999-point caps, one 0.1 m p2p ICP and Acc/Comp/NC1/NC2.
  Actual criterion/Open3D regression is exact with OMP_NUM_THREADS=1. Two-thread ICP
  showed ~1e-16 floating reduction variation; the test is not weakened to hide it.
- Mean NC=(NC1+NC2)/2, sequence metrics averaged without frame weighting. VGGT-Long
  uses point-head world_points; ours_v4 uses depth-unprojection points. This comparison
  is not a pure batching ablation.

## Timing and artifacts

Each sequence records preprocessing, pure VGGT forward, packing/transfer, prediction
conversion, overlap stitching and point-map construction. Reconstruction total is
the sum of these required stages, excluding all diagnostic export, GT loading,
protocol evaluation and plotting. Every GPU stage is synchronized before/after.
Protocol evaluation includes GT loader and metrics, separately timed.
CUDA allocated/reserved peaks reset per sequence with weights resident. Per-sequence
RSS is sampled at 100 ms; process cumulative high-water RSS is also retained and
clearly distinguished. Actual GPU UUID is saved in manifest and every sequence.

Saved artifacts: manifest/source hashes/checkpoint identity, sequences.jsonl,
summary.json, frame IDs/paths, windows/packed groups/ownership, per-window cameras,
global trajectory, every Sim3/residual/inlier record, timing diagnostics, logs and
VRAM CSV. Dense local outputs are saved for the equivalence gate; formal evaluation
uses dense predictions in memory and keeps compact diagnostics, avoiding multi-GB
prediction dumps for every test sequence. No extra ICP or GT-assisted failure recovery.
