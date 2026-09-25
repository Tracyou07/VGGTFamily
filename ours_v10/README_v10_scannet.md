# v10 ScanNet fixed-input entry

This adapter uses the frozen `scene0000_00` 1000-frame `inputs.pt` in place. It
never extracts or preprocesses images again. It permits only prefix lengths
100, 300, 500 and 1000, with 60-frame windows and 10-frame overlap (2, 6, 10
and 20 windows). Every length gets a fresh forward pass because the v10 camera
bank depends on all windows in that run. The Scene20 entry remains separate.

The pinned input SHA-256 is
`0c98205a8acef8558fbf522de364df7bd0a9e482f0b27b9ca54a08690085610b`;
the checkpoint SHA-256 is
`f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
The predictor validates the full file, ordered frame IDs, tensor shape and
dtype, checkpoint, and every requested GT pose before GPU execution. GT poses
are discarded after preflight. It records the prefix tensor hash and every
window prediction hash. The CPU alignment entry verifies them before invoking
the existing v9 sparse alignment code. No point cloud is exported.

Run from `/home/ubuntu/yjh/feedforwardreconstruct/ours_v10` in the
`/home/ubuntu/anaconda3/envs/vggt-gx` environment. Choose a genuinely idle
physical H20 after checking `nvidia-smi`, active jobs, and `df -h / /data`.
Set `GPU` and a new `RUN` for **each length**. Never reuse a completed run
directory. The files belong on the root filesystem under
`/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments`; do not copy
`inputs.pt` or place these outputs on `/data`.

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v10
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
: "${GPU:?set GPU to an idle physical H20 index}"
FRAMES=100  # repeat as separate runs with 300, 500, and 1000
RUN=/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/$(date -u +%Y%m%dT%H%M%SZ)_scannet_f${FRAMES}
for MODE in overlap_correspondence camera_global_overlap; do
  env -u CUBLAS_WORKSPACE_CONFIG CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" -u -m experiments.ours_v10.predict_scannet \
      --frames "$FRAMES" --mode "$MODE" --gpu "$GPU" \
      --backend-profile native_vggt --output-root "$RUN" \
      2>&1 | tee "${RUN}_${MODE}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
  "$PY" -u -m experiments.ours_v10.align_scannet \
    --run-root "$RUN" --mode "$MODE" --output "$RUN/alignment/$MODE" \
    2>&1 | tee "${RUN}_${MODE}_align.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done
"$PY" -m experiments.ours_v10.report_scannet \
  --run-root "$RUN" --frames "$FRAMES" --output "$RUN/report"
```

The example is the 100-frame gate. The same commands with `FRAMES=300`,
`500`, or `1000` launch independent forwards, not truncated predictions.
Prediction and alignment processes write `FAILED.json` on failure and never
overwrite an existing mode directory. Stop at any failure. The report checks
that both modes used the same input prefix and noncommunication settings, then
records same-length ATE, adjacent and boundary errors, per-edge scale/fallback,
forward and sparse-stitch time, allocated/reserved GPU peaks, and CPU RSS.
Reserved memory at or above 24 GiB is explicitly marked as missing the goal.

For the CPU gate:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v10
/home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest discover -s tests/ours_v10 -v
```

Each full 1000-frame v8 prediction set occupied about 12 GiB on this
filesystem. Budget about 2.4 GiB for the 100-frame two-mode gate and roughly
46 GiB for both modes at all four lengths, plus alignment files and slack.
Recheck actual free space and other active jobs before each run. Do not delete
old artifacts to make space.
