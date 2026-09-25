# v9 sparse point and camera stitching experiment

This branch isolates the stitching experiment from the v8 model. It reads v8
`windows/*/local.npz` only. It never loads model weights or invokes a GPU
forward pass. `point_camera_joint` uses the original v8 `AlignmentStitcher`;
`sparse_point_camera_joint` changes only the point pairs supplied to the Long
initializer and the existing v8 joint optimizer. Window ownership and the
whole-trajectory GT Sim(3) evaluation are unchanged.

For each shared original frame, the selector splits the original point-map
pixel grid into 16 × 16 cells. It keeps at most one same-pixel B→A pair in
each cell: the finite pair above the original confidence threshold with the
largest product of the two confidences. A row-major tie break makes the
choice deterministic. It records frame ID, row, column, both confidences and
a SHA256 of the canonical ordered index list. At most 256 pairs per frame
enter both the unchanged Long weighted Huber IRLS initializer and the v8
point-plus-camera Huber optimizer.

Predeclared sparse acceptance gates require at least 32 populated cells and
all four image quadrants on every shared frame, rank at least two in source
and target 3D points, a finite positive geometry scale, optimizer success,
final scale between 0.1 and 10, and no objective increase above 1e-8.
Controlled sparse failures are recorded and retried with the unchanged full
`point_camera_joint`. A failure of that fallback fails the edge. No GT or
diagnostic holdout pixel participates in selection, fitting or fallback.

The CPU runner loads one frozen window prediction at a time, writes each
edge's transform, point counts, stage times, process RSS and status, then
evaluates the finished trajectory. Prediction loading, alignment, diagnostic
residual checks, export and evaluation have separate timers. The other-pixel
diagnostic samples pixels distinct from sparse selected pixels; it is an
out-of-fit diagnostic only when sparse fitting succeeded without fallback.
For the full baseline those same pixels participated in full fitting.

## CPU tests

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v9
PYTHONPATH=. /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest discover -s tests/ours_v9 -p 'test_*.py' -v
```

## Frozen 1000-frame experiment

Run once for each prediction set and mode, always with a new output path:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v9
FROZEN=/home/ubuntu/yjh/feedforwardreconstruct/ours_v8_experiments/20260923T074952Z_scene0000_00_f1000_w60_o10
SET=independent # or overlap_correspondence
MODE=sparse_point_camera_joint # or point_camera_joint
OUT=/home/ubuntu/yjh/feedforwardreconstruct/ours_v9_experiments/NEW_UNIQUE_ID/${SET}_${MODE}
PYTHONPATH=. /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -u -m experiments.ours_v9.compare_frozen_sparse \
  --predictions "$FROZEN/$SET/windows" \
  --source-manifest "$FROZEN/$SET/run_manifest.json" \
  --prediction-set "$SET" --mode "$MODE" --output "$OUT"
```

`--output` must not exist. The runner writes `COMPLETE.json` only after all
19 edges and whole-trajectory evaluation succeed; exceptions leave
`FAILED.json` and edge progress in that new directory.
