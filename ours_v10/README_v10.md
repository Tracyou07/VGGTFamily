# ours_v10: global camera bank plus v9 overlap correspondence

`ours_v10` is an isolated branch/worktree based on v9 commit
`db7d4df5fc5905e7175bce60aefd4622ecc98ab6`. The v9 directory,
algorithm and old results are unchanged. Modes are `independent`,
`camera_only`, `overlap_correspondence` and `camera_global_overlap`.

Only global attention differs. A camera Query reads all local K/V and every
other window's camera K/V. An overlap patch Query reads all local K/V plus
its adjacent same-frame, same-pixel patch K/V. Other patch and register
Queries read local K/V only. Both communicating groups use one joint
softmax across their permitted keys. All windows project Q/K/V before any
same-layer global update. No overlapping camera instances are merged or
averaged; each output manifest records duplicate window-frame instances
and remote bank counts. Frame attention, per-window reference, v8 heads,
front-window ownership and v9 sparse point-camera Sim(3) are unchanged.

The v10 `vggt/v10/model.py` differs from `vggt/v8/model.py` only in its
module docstring; its relative scheduler import selects v10. The CPU
alignment runner differs from v9 only by admitting the two new mode names.
No v8 or v9 source file is edited in its original worktree.

## Tests and small GPU operator check

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v10
PYTHONPATH=. /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -m unittest discover -s tests/ours_v10 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -u -m experiments.ours_v10.gpu_operator \
  --backend-profile native_vggt --gpu 4 \
  --output /home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/NEW_UNIQUE_gpu_operator
```

CPU dense-reference tolerances are FP64 `1e-10` and FP32 `2e-5` for
absolute and relative differences. The separate BF16 GPU operator gate
declares `atol=rtol=0.03` before execution. Its profiler is diagnostic,
not a formal timing result. Check H20 identity, free GPU, disk and active
jobs before every GPU run; the CLI refuses a busy or non-H20 GPU.

## Fixed 100-frame gate

Reuse the first 100 preprocessed frames of the frozen Scene20 Clone
`inputs.pt`. Run four modes in separate fresh GPU processes and new output
directories:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v10
FROZEN=/home/ubuntu/yjh/feedforwardreconstruct/ours_v9_experiments/20260923T144250Z_vkitti131_scene20_clone_predictions
GATE=/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/NEW_UNIQUE_scene20_clone_f100_gate
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
for MODE in independent camera_only overlap_correspondence camera_global_overlap; do
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. "$PY" -B -u -m experiments.ours_v10.predict \
    --backend-profile native_vggt --frozen-input-root "$FROZEN" \
    --output-root "$GATE" --mode "$MODE" --gpu 4 --frames 100 || break
done
PYTHONPATH=. "$PY" -B -u -m experiments.ours_v10.compare_raw \
  --prediction-root "$GATE" --output "$GATE/raw_prediction_differences.json"
```

For geometry gate, call `experiments.ours_v10.compare_frozen_sparse` once
per mode with its `windows`, `run_manifest.json`, `--prediction-set MODE`,
`--mode sparse_point_camera_joint`, `--vkitti-raw-root` pointing to the
Virtual KITTI 1.3.1 extracted root, `--vkitti-condition clone`, and a
fresh `--output`. GT is loaded only after stitching finishes.

## Full Scene20 predictions and alignment

The three v9 frozen roots already contain exactly 837 ordered frames and
must not be rewritten. For each `CONDITION` in `clone rain fog`, run only
the two **new** modes, serially on the same free H20:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v10
CONDITION=clone  # repeat with rain and fog
FROZEN=/home/ubuntu/yjh/feedforwardreconstruct/ours_v9_experiments/20260923T144250Z_vkitti131_scene20_${CONDITION}_predictions
OUT=/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/NEW_UNIQUE_${CONDITION}_predictions
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
for MODE in camera_only camera_global_overlap; do
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. "$PY" -B -u -m experiments.ours_v10.predict \
    --backend-profile native_vggt --frozen-input-root "$FROZEN" \
    --output-root "$OUT" --mode "$MODE" --gpu 4 --frames 837 || break
done
```

For each new prediction set, run CPU sparse alignment (new output per mode):

```bash
RAW=/data/yjh/share/datasets/Virtual_KITTI_1.3.1/extracted
ALIGN=/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/NEW_UNIQUE_${CONDITION}_alignment
for MODE in camera_only camera_global_overlap; do
  PYTHONPATH=. "$PY" -B -u -m experiments.ours_v10.compare_frozen_sparse \
    --predictions "$OUT/$MODE/windows" \
    --source-manifest "$OUT/$MODE/run_manifest.json" \
    --prediction-set "$MODE" --mode sparse_point_camera_joint \
    --vkitti-raw-root "$RAW" --vkitti-condition "$CONDITION" \
    --output "$ALIGN/$MODE" || break
done
```

Use `experiments.ours_v10.build_report --source-spec SPEC.json --output
NEW_UNIQUE_REPORT` to audit old and new manifests together and export
mode/edge CSVs. The completed example and its exact source spec are under
`/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/20260924T021000Z_scene20_report`
and `20260924T021000Z_scene20_source_spec.json`.

The first full validation is a **research result**: its new modes have
~26.71 GiB peak reserved, above the requested 24 GiB budget, and Rain
accuracy regresses. Do not promote either new mode as a default or claim a
stable speedup from one timing run.
