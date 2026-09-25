# Virtual KITTI 1.3.1 Scene20 stress-test entrypoints

These opt-in scripts extend `ours_v9` without changing the v9 sparse solver or
the v8 forward implementation. Conditions are `clone`, `rain`, and `fog` under
the Virtual KITTI **1.3.1** RGB and extrinsics layout. The official
`eval/virtual_kitti/src/virtual_kitti_eval` package validates the source,
inverts its world-to-camera GT into camera-to-world, and computes one
whole-trajectory proper positive-scale Sim(3) ATE. GT is loaded for CPU
evaluation only after `stitcher.finish()`. It is not serialized into
`inputs.pt` and is never passed to the model, point selector, fit, or fallback.

The full input is the original continuous `00000`–`00836` (837 frames).
`window_size=60, overlap=10` yields 17 windows, 16 edges, with last window
`[800,837)`. No frame is duplicated to reach 1000. The prediction script
preprocesses once and launches independent and overlap-correspondence in
separate fresh processes against the same saved CPU tensor. It saves only
`windows/NNNN/local.npz` fields required by alignment: `frame_ids`, `c2w`,
`intrinsics`, `depth`, `world_points`, and `world_points_conf`. Input tensor,
checkpoint, source and prediction hashes, backend settings, GPU UUID and
allocated/reserved peaks are in manifests. These GPU peaks describe prediction,
**not** a memory benefit of CPU sparse alignment.

The CPU entry verifies both prediction sets and their file hashes, then runs
each with full `point_camera_joint` and sparse `sparse_point_camera_joint`.
Per-edge JSON/CSV include correspondence counts and spatial coverage, fallback
status/reason, transforms, residuals, boundary errors, stage times, and
incremental RSS sampled during `stitcher.add()` at 5 ms intervals. The
process-wide `ru_maxrss` peak is a separate field. Sampling can miss short RSS
spikes. Alignment output writes `FAILED.json` on failure and never edits the
frozen predictions.

Run from the v9 repository root in the `vggt-gx` environment. Each `OUT` must
be a **new path** on `/home/ubuntu` (not the nearly full `/data` partition).
Check H20 identity, free GPU, disk and active processes first. The prediction
entry itself refuses a busy or non-H20 GPU.

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v9
OUT=/home/ubuntu/yjh/feedforwardreconstruct/ours_v9_experiments/UNIQUE_scene20_clone_predictions
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -u -m experiments.ours_v9.vkitti_predict \
  --condition clone --gpu 4 --output "$OUT"
```

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v9
PRED=/home/ubuntu/yjh/feedforwardreconstruct/ours_v9_experiments/UNIQUE_scene20_clone_predictions
OUT=/home/ubuntu/yjh/feedforwardreconstruct/ours_v9_experiments/UNIQUE_scene20_clone_alignment
PYTHONPATH=. /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -u -m experiments.ours_v9.vkitti_align \
  --prediction-root "$PRED" --condition clone --output "$OUT"
```

Change only `--condition` and unique output paths for `rain` and `fog`.
For an explicit interface smoke test, the prediction command accepts
`--smoke-frames 65`; add `--smoke` to the CPU alignment command. Smoke outputs
are labelled as shortened tests and cannot be mistaken for full Scene20 results.

CPU regression gate:

```bash
PYTHONPATH=. /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest discover -s tests/ours_v9 -p 'test_*.py' -v
```
