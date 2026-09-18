# ScanNet evaluation repository design

The user explicitly requests reuse of FastVGGT's ScanNet evaluation script. The standalone Git repository at `/home/ubuntu/yjh/feedforwardreconstruct/eval/scannet` therefore keeps that script and its metric implementation as the reference, and repairs infrastructure around it. Supported inference implementations are VGGT original, VGGT star, FastVGGT, StreamVGGT, native VGGT-Long, VGGT-SLAM and VGGT-Omega.

## Reference and protocol

Preserve a checksummed source snapshot of `eval/eval_scannet.py`, `vggt/utils/eval_utils.py`, `eval/scannet_50.yaml` and its license from the existing FastVGGT checkout. Runtime metric helpers are reused from that source, with model-specific imports made lazy so another backend cannot import the wrong `vggt` namespace. The main `eval_scannet.py` entry point keeps familiar upstream flags and processing stages, delegating safe data/model orchestration to the package.

Keep the original metric keys and numerical convention: `chamfer_distance`, `ate`, `are`, `rpe_rot`, `rpe_trans`, `inference_time_ms`. The reference uses aligned evo trajectory errors, its world-to-camera trajectory representation, independent bounding-box diagonal scale/center alignment for point geometry, deterministic seed33 caps of100000 points,5cm voxel downsampling, and summed bidirectional mean distances clipped at0.5m. Document these conventions explicitly as the FastVGGT reference protocol; do not silently replace them with a new shared Sim3 protocol or claim these are conventional camera-center ATE scores. Tests must compare real numerical outputs with the untouched reference on nontrivial synthetic predictions, including scale/orientation mismatch. Reject invalid inputs before reference calls, and reject missing/nonfinite metric outputs afterward.

## Data

Use the existing50-scene list and raw `.sens` plus official GT PLY under `/data/yjh/share/datasets/ScanNet`. Prepare a new cache at `prepared_scannet50_v1`, preserving all valid-pose RGB frames by default, original IDs, camera-to-world TXT poses and calibration. Export `color/<id>.jpg` and `pose/<id>.txt` for direct upstream compatibility. Do not reconstruct GT from low-resolution depth or create projected-depth aliases. Runtime selection follows upstream `build_frame_selection` exactly once, with default input_frame1000. A smaller profile records its selected IDs. Exported encoded RGB, TXT and GT assets have complete hashes/manifests; truncation, partial data and corruption cannot pass validation or resume.

## Backend contract and isolation

Each backend emits finite Nx3 predicted-world points and one rigid4x4 c2w pose per requested original frame ID, inference time and peak GPU memory. Convert c2w to the reference's w2c representation only at the evaluator boundary. Do not reinterpret the native algorithm: Long executes its chunk/loop pipeline; SLAM its solver. Model-specific filtering is explicit. FastVGGT uses the upstream depth confidence threshold1.0 and preserves all finite depth points until the upstream evaluator applies its own point cap; optional earlier caps must be recorded as a changed profile.

Use existing vggt-gx for VGGT variants/Stream/Omega/Long and monst3r for SLAM. Put additional Long dependencies in repository-local `.runtime/long_deps`. Reference existing VGGT/DINO/SALAD weights, leave external model repositories and shared environments unchanged. Use separate processes for conflicting `vggt` implementations. Serialize bounded GPU smoke runs after checking dynamic resources and preserve existing jobs.

## Result integrity

Validate requested inputs before model allocation. Empty scenes, missing/mismatched frames, invalid transforms, degenerate trajectories, missing GT and nonfinite scores fail visibly. Zero valid scenes never becomes a zero-score success. Preserve successful resumed scenes in aggregate; report expected/success/failed counts and require the whole requested run to complete for exit0. Bind outputs to model, checkpoint, source, upstream protocol, selected frames and data hashes. Resume only exact matching provenance, and never overwrite unrelated/stale outputs.

## Delivery

Provide prepare/verify/doctor/run/aggregate commands plus familiar FastVGGT entrypoint flags. Deliver all50 scenes prepared and checked, CPU regression/parity tests, seven actual one-scene/eight-frame GPU smoke results when resources allow, committed source and concise documentation. Smoke validation is executable integration evidence, not a final ScanNet50 benchmark. Do not silently launch the complete350-run matrix. Record all changes relative to upstream and all resource limitations.
