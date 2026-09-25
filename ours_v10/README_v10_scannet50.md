# v10 ScanNet-50 adapter

This is a separate, opt-in evaluation entry. It leaves the v10 model, all
attention modes, the v9 sparse point-plus-camera Sim(3), the frozen ScanNet
prefix entry, and baseline outputs unchanged. The protocol ID is
`fastvggt_scannet_evo132`; scoring calls the existing
`eval/scannet/scannet_eval.fastvggt_eval.evaluate_prediction` implementation.
It reports that protocol's six principal fields and all 13 scene fields.
`inference_time_ms` is v10 forward including CPU prediction transfer plus
sparse stitching including local prediction reads. Preprocessing, NPZ export
and GT scoring are recorded separately and excluded from that value.

The selected frames come from the prepared valid-pose frame list using
`scannet_eval.sens.sample_frame_ids`, which preserves the first frame and the
FastVGGT stride rule. A CPU audit compares all 50 scenes and 100/300/500/1000
budgets against **complete** VGGT* scene results before a model run. If a
scene has fewer valid frames than its budget, all valid frames are used. GT is
read to validate the scene and, after stitching, by the reference evaluator;
it is never passed to v10 forward, sparse point selection or fallback.

## One-scene gate

Run these commands on H20. Select a physical GPU that is idle at launch; the
worker refuses an occupied card. `OUT` belongs on `/data` and `SCRATCH` on
the root filesystem. Each command produces one new `(scene, budget)` run.
No command below launches the 50-scene batch.

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v10
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
GPU=0  # replace with an idle physical H20 index
TAG=$(date -u +%Y%m%dT%H%M%SZ)_v10_scannet50_gate
OUT=/data/yjh/output/vggt/$TAG
SCRATCH=/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/${TAG}_scratch
hostname
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
df -h / /data
$PY -m unittest tests.ours_v10.test_scannet50_contract tests.ours_v10.test_scannet50_pipeline -v
$PY -m experiments.ours_v10.scannet50 audit-selection --output /home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/${TAG}_selection_audit.json
for N in 100 1000; do
  $PY -u -m experiments.ours_v10.scannet50 run \
    --scene scene0000_00 --frames "$N" --gpu "$GPU" \
    --output-root "$OUT" --scratch-root "$SCRATCH"
done
```

For a stopped or failed stage, use the same `OUT`, `SCRATCH`, scene, budget,
mode and GPU with `--resume`. Successfully completed stages are verified and
reused. An interrupted stage is retried in a new attempt directory; its older
attempt stays available until complete scene scoring confirms cleanup.

```bash
$PY -u -m experiments.ours_v10.scannet50 run \
  --scene scene0000_00 --frames 1000 --gpu "$GPU" \
  --output-root "$OUT" --scratch-root "$SCRATCH" --resume
```

## Experiment-group 50-scene command

**Do not run this during adapter development.** After the single-scene gate,
choose a new unique `TAG`, recheck GPU and disk, then run serially. A failed
scene stops the loop and retains its scratch files; rerun the same command
with `--resume` after diagnosis. Per-scene scratch `inputs.pt` and `local.npz`
are unlinked only after the 13 metrics, result, edge transforms, trajectory and
per-scene `summary.complete=true` have been persisted and checked. A receipt
lists every removed path, hash and byte count. The output root retains no
large point cloud file.

```bash
set -euo pipefail
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v10
PY=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
GPU=0  # set to a currently idle physical H20
TAG=$(date -u +%Y%m%dT%H%M%SZ)_v10_scannet50
OUT=/data/yjh/output/vggt/$TAG
SCRATCH=/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments/${TAG}_scratch
for N in 100 300 500 1000; do
  while IFS= read -r SCENE; do
    [[ -z "$SCENE" || "$SCENE" == \#* ]] && continue
    /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -u -m experiments.ours_v10.scannet50 run \
      --scene "$SCENE" --frames "$N" --gpu "$GPU" \
      --output-root "$OUT" --scratch-root "$SCRATCH" --resume
  done < /home/ubuntu/yjh/feedforwardreconstruct/eval/scannet/configs/scannet50.txt
  "$PY" -m experiments.ours_v10.scannet50 aggregate --frames "$N" --output-root "$OUT"
done
```

The aggregate includes only verified `COMPLETE.json` scenes. Baseline averages
are cited only when their own `summary.complete=true`; StreamVGGT 500/1000
remain explicitly missing. An incomplete v10 aggregate has
`summary.complete=false` and a nonzero command exit. Never convert a missing
method or scene into a numeric score.
