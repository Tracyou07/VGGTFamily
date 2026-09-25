# ours_v4: independent packed VGGT* windows

Original model: cc1d8ac15861aea54d14961653cd340e7d984f29. Both model trees are snapshots; `reference/vggt` is selected in a separate process for the baseline. Model and attention code are unmodified. Every window has its own reference frame and cloned image instance. Same-length windows are batched; the short tail is separate. No feature cache, communication, new parameters, training or optimizer.

Stitching: pinned ours_v3 d54c6e2c6e8e0491d04bd7d4105e455217aeb884. Geometry, RANSAC seed/thresholds and first-occurrence deduplication are identical for both paths. GT is read only after stitching. Saved poses are c2w, depth is camera z. Positive Sim3 scale applies to depths and centers, never camera rotations.

Checkpoint: /data/yjh/share/pretrained/VGGT-1B/model.safetensors
SHA256: f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/ours_v4
PYTHON=/home/ubuntu/anaconda3/envs/vggt-gx/bin/python
OMP_NUM_THREADS=2 "$PYTHON" -B -m unittest discover -s tests/ours_v4 -v
# Check identity, nvidia-smi, active jobs and df -h /data first; use an idle GPU.
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2  "$PYTHON" -B -u -m experiments.ours_v4.validate  --output /data/yjh/output/vggt/ours_v4/${RUN_ID}
```

The ordered campaign runs FP32 and BF16 reference-repeat and batch=2 equivalence on 55 frames (30,30,15), comparing full encoder tensors, all cached head features, depth, confidence, intrinsics and poses. Tolerances are fixed in configs/v4_validation.json. Any failure stops the campaign; do not relax thresholds or start downstream experiments. Only after both gates pass does it run 100-frame reference and packed BF16 diagnostics serially with the same saved input tensors. New output directories are mandatory.

Outputs include complete tensor captures, per-window local predictions, provenance, error reports, logs and resource measurements. Diagnostic runs additionally produce stitching transforms, residuals, global trajectory, PLY and GT comparison. COMPLETE is written last. CPU RSS is process peak; GPU allocated/reserved include weights and full heads. Packing increases concurrent memory; no 500/1000-frame claim. No frame feature caching. Dispatcher-selected SDPA uses the original implementation; enabled backends are recorded rather than guessing the selected kernel.
