# Virtual KITTI 1.3.1 data and evaluation contract

Primary evidence: [NAVER LABS Europe, Virtual KITTI 1](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/), checked 2026-09-15. The Downloads, Rendered RGB frames, Camera pose, and camera-coordinate sections establish:

- Release 1.3.1 uses worlds 0001, 0002, 0006, 0018 and 0020.
- RGB is `vkitti_1.3.1_rgb/<world>/<variation>/%05d.png`; frames begin at zero.
- Extrinsics are `vkitti_1.3.1_extrinsicsgt/<world>_<variation>.txt`.
- Each extrinsics row is the frame index followed by 16 row-major 4x4 matrix coefficients. The matrix maps world points to camera coordinates.
- This release is monocular. The published camera matrix is `[[725,0,620.5],[0,725,187],[0,0,1]]`, with right/down/forward camera axes.

No archive was downloaded or extracted to establish this contract.

## Accepted layout

```text
raw_root/
  VERSION                               # optional explicit version, e.g. 1.3.1
  vkitti_1.3.1_rgb/
    0001/clone/00000.png
    0001/clone/00001.png
    ...
  vkitti_1.3.1_extrinsicsgt/
    0001_clone.txt
    ...
```

Official version-bearing top-level directory names and an optional plain `VERSION` file are filesystem version markers. All observed markers must agree. These checks establish release/layout consistency, not cryptographic authenticity of a downloaded dataset. A `VERSION` file alone cannot authorize a guessed layout; unknown structure returns `DATASET_LAYOUT_UNVERIFIED`.

The parser requires a 17-column whitespace-separated header beginning with `frame`, then ordered rows of decimal frame index and 16 numeric coefficients. Descriptive matrix header names are not interpreted; their documented positions determine the matrix. A camera-ID column, a 2.0.3 `SceneXX/.../Camera_0` layout, and Camera_1 selection are rejected.

The adapter maps canonical `Scene01/Clone` to `0001/clone` only at the raw-data boundary. Main evaluation is Scene01/02 crossed with Clone, Fog, Morning, Overcast, Rain and Sunset. Scene06/18/20 can be prepared only when listed explicitly in the sequence config; they do not become extra columns in the main tracking table.

Version mismatch and partial RGB archives are checked before layout discovery. Text GT alone is never ready. Every RGB PNG must have a contiguous five-digit filename, pass PNG integrity verification and full pixel decoding, contain RGB pixels, and share the sequence's dimensions. GT row order and counts must match exactly. Extrinsics must be finite, rigid, homogeneous, and orientation preserving. The adapter inverts each w2c transform to c2w.

## Prepared state and inputs

Preparation writes `prepared_root/SceneXX/Condition/{manifest.json,poses_c2w.npy}`. RGB stays in its original location; source records contain relative paths, byte sizes, and SHA-256 values for every image, GT file and explicit VERSION marker. A temporary sibling directory is renamed into place only after source validation. Existing prepared state is verified and never overwritten.

Verification fully revalidates and decodes raw data, checks canonical JSON content, file integrity, path containment, normalized poses, and exact agreement with raw GT. It rejects stale sources and rehashed normalized-pose tampering. The optional `verify_hashes` argument is retained for future caller compatibility; verification always checks content.

`PreparedSequence.timestamps_s` is `None`: this contract supplies frame indexes, not measured timestamps. Intrinsics retain the documented native camera matrix as evaluation metadata. No resize, artificial timestamp or model inference occurs during preparation.

`make_backend_request` exposes only ordered `frame_ids` and `image_paths`. Ground-truth poses and intrinsics remain on the evaluator side. Depth, semantic, instance and optical-flow data are neither required nor read nor exposed.

## Metrics, results and resume

ATE uses exact frame-ID matching and an independent local NumPy Sim(3) implementation, evaluating camera-center translation RMSE in metres after positive-scale, proper-rotation alignment. At least three non-collinear positions are required. Degenerate, nonfinite and full-rank reflection cases fail explicitly.

Each scene/condition has separate metrics/result files. The run manifest binds config and sequence files, frame order, prepared manifests, model source tree, checkpoint, interpreter, selected device, timeout, command, evaluator source tree and metric protocol. Resume requires successful matching result/metrics pairs, matching hashes, and unchanged live sources. No backend is loaded to validate provenance.

JSON is finite, duplicate-key rejecting and atomically replaced. Metrics are written before the terminal result, which acts as their content-bound commit marker. Aggregation reports missing, failed or stale pairs as incomplete; it never fills them with zero. Every main condition has an independent table column; there is no tracking-table average column. Optional configured pairs remain independent in machine-readable output.

The resource table retains model, scene/condition, input-frame count, inference seconds, peak allocated VRAM in MiB and status. There is no synthetic resource measurement in this task.
