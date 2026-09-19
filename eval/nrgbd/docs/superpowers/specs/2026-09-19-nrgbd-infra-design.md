# NRGBD Unified Evaluation Infrastructure Design

**Date:** 2026-09-19  
**Status:** Draft for review  
**Target:** `/home/ubuntu/yjh/feedforwardreconstruct/eval/nrgbd`

## 1. Goal

Build a standalone, reproducible NRGBD evaluation package for five reconstruction systems:

1. Original VGGT
2. VGGT-Long
3. StreamVGGT
4. VGGT-SLAM
5. VGGT-Ω

All systems must use one data loader, one scene selection policy, one scoring implementation, and one result schema. The evaluation protocol must reproduce FastVGGT's published NRGBD setup. The first delivery builds and verifies the infrastructure without starting real GPU evaluation.

## 2. Scope

The package includes:

- NRGBD data discovery and validation
- FastVGGT-compatible frame selection and preprocessing
- a common prediction contract
- five model adapters
- FastVGGT-compatible geometry scoring
- deterministic sampling and aggregation
- per-scene atomic output and resume
- inference-time, wall-time, CUDA-memory, and external GPU-memory reporting
- H20 configuration and launch scripts
- CPU-only unit and integration tests
- documentation and GitHub publication under `eval/nrgbd`

The package excludes:

- dataset downloads or rewriting
- checkpoint downloads
- model training
- real GPU evaluation during infrastructure development
- generated metrics, point clouds, caches, logs, and weights from source control
- the NRGBD `archives` directory from evaluation

## 3. Protocol source and identity

The authoritative reference is:

`/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/reference/FastVGGT-main/eval/eval_7andN.py`

The source dataset is:

`/data/yjh/share/datasets/NRGBD`

The protocol identifier is `fastvggt_nrgbd_kf10_v1`. Results are comparable only when this identifier and the recorded protocol fingerprint match.

The implementation preserves these FastVGGT choices:

- test split
- one full video per scene
- every tenth input frame (`kf=10`)
- input resolution `(518, 392)`
- center `224 × 224` scoring crop
- GT construction from NRGBD depth, camera pose, and intrinsics
- at most 999,999 predicted points and 999,999 GT points per scene
- point-to-point ICP with a 0.1 m correspondence threshold
- accuracy, completion, and bidirectional normal consistency
- mean aggregation across scenes

The original FastVGGT code uses filesystem order and unseeded random point sampling. The new package sorts scene and frame identifiers and uses a recorded seed for deterministic sampling. This preserves the formula and sampling limit while making reruns reproducible. The result metadata explicitly records this deterministic clarification.

## 4. Dataset contract

The nine expected evaluation scenes are:

- `breakfast_room`
- `complete_kitchen`
- `green_room`
- `grey_white_room`
- `kitchen`
- `morning_apartment`
- `staircase`
- `thin_geometry`
- `whiteroom`

Each scene must contain:

- `images/img{frame_id}.png`
- `depth/depth{frame_id}.png`
- `poses.txt`

The loader intersects RGB, depth, and pose-valid frame IDs, sorts numeric IDs, then takes `ids[::10]`. It must reject:

- missing expected scenes
- extra selected scenes unless explicitly requested
- duplicate or nonnumeric frame IDs
- missing RGB/depth pairs
- truncated pose matrices
- nonfinite valid poses
- empty selections
- a frame count that would change because a required modality is absent

The camera intrinsics are fixed to the FastVGGT values:

```text
fx = fy = 554.2562584220408
cx = 320
cy = 240
```

Depth PNG values are millimetres, converted to metres. Depth below 0.001 m or above 10 m is invalid. Camera poses receive the same OpenGL-to-OpenCV Y/Z-axis conversion as FastVGGT.

The `check` command is read-only. It reports selected frame IDs, counts, missing inputs, pose validity, sizes, and a dataset fingerprint without loading model weights or allocating CUDA memory.

## 5. Preprocessing and ground truth

RGB is resized to the depth image dimensions before the FastVGGT crop/resize operation. The image, depth, intrinsics, validity mask, and pose undergo the same geometric transform.

The scorer reconstructs GT world points from the transformed depth, intrinsics, and pose. Model adapters must not receive GT depth, GT points, or GT poses as inference inputs. GT data crosses the boundary only inside the scorer.

For scoring, every frame is center-cropped to 224 × 224 exactly as in the reference. Invalid depth and padding are excluded.

## 6. Common prediction contract

Every adapter implements:

```python
class Backend(Protocol):
    name: str
    def doctor(self) -> DoctorReport: ...
    def predict(self, scene: SceneInput, work_dir: Path) -> ScenePrediction: ...
```

`ScenePrediction` contains:

- `scene_id`
- ordered `frame_ids`
- per-frame predicted world points or depth plus camera-to-world pose and intrinsics
- per-frame valid masks
- optional confidence used only when the protocol configuration enables it
- synchronized model-forward seconds
- total adapter seconds
- peak CUDA allocated bytes
- peak CUDA reserved bytes
- model/checkpoint/source fingerprints
- adapter-specific diagnostics

The runner validates finite arrays, shapes, frame count, exact frame ordering, rigid poses where applicable, and prediction provenance before scoring.

No adapter may implement its own metric, ICP, GT loader, scene selection, or summary aggregation.

## 7. Model adapters

### Original VGGT

Loads the existing VGGT-1B checkpoint and produces camera, depth/world-point, validity, and confidence outputs using the original VGGT inference path.

### VGGT-Long

Uses the existing VGGT-Long project and its long-sequence prediction path. Windowing, alignment, and fusion remain model behavior; the adapter converts the final output to the common contract.

### StreamVGGT

Uses the existing StreamVGGT checkpoint and streaming state. The adapter feeds frames in selected order, finalizes the scene once, and exports one prediction for every selected frame.

### VGGT-SLAM

Uses the existing VGGT-SLAM pipeline, including its loop-closure behavior. The adapter exports its final globally consistent scene reconstruction. Any pose convention conversion is explicit and tested.

### VGGT-Ω

Uses the existing VGGT-Ω checkpoint and inference code. The adapter normalizes output resolution, pose convention, and point/depth representation to the common contract.

Each adapter's `doctor` verifies source root, checkpoint path, Python dependencies, model source fingerprint, and device capability without running a scene. Backend construction remains allocation-free until `predict`.

## 8. Scoring

For each scene:

1. Build predicted and GT point clouds from the common prediction and scorer-owned GT.
2. Apply the FastVGGT scale/shift-invariant alignment used by `Regr3D_t_ScaleShiftInv(..., gt_scale=True)`.
3. Apply validity masks and the center crop.
4. Deterministically sample each cloud to at most 999,999 points using the configured seed and scene ID.
5. Run Open3D point-to-point ICP with identity initialization and threshold 0.1 m.
6. Estimate normals after ICP.
7. Compute:
   - `acc`: predicted-to-GT mean nearest-neighbour distance
   - `acc_med`: predicted-to-GT median distance
   - `comp`: GT-to-predicted mean nearest-neighbour distance
   - `comp_med`: GT-to-predicted median distance
   - `nc1` and `nc1_med`: absolute normal dot product in the accuracy direction
   - `nc2` and `nc2_med`: absolute normal dot product in the completion direction
   - `nc = (nc1 + nc2) / 2`
   - `nc_med = (nc1_med + nc2_med) / 2`

Empty, nonfinite, or degenerate point clouds fail the scene with a clear diagnostic; they never produce zero-filled metrics.

The summary is the unweighted arithmetic mean across successfully completed expected scenes. A canonical complete summary requires all nine scenes. Partial summaries are labelled `partial` and list missing/failed scenes.

## 9. Results, provenance, and resume

Directory layout:

```text
nrgbd/
├── README.md
├── pyproject.toml
├── configs/h20.json
├── docs/
├── reference/
├── scripts/
├── src/nrgbd_eval/
└── tests/
```

Runtime output layout:

```text
results/<model>/<run_id>/
├── run_manifest.json
├── environment.json
├── scenes/<scene_id>/
│   ├── prediction_metadata.json
│   ├── metrics.json
│   └── COMPLETE
├── summary.json
├── summary.csv
└── resources/
```

Large point clouds are optional debug artifacts and disabled by default.

A scene is complete only after prediction metadata and metrics are durably written and an atomic `COMPLETE` marker is created. Resume skips only scenes whose protocol, config, model source, checkpoint, input, and implementation fingerprints match. Failed staging directories are retained separately for diagnosis and never counted as complete.

Existing nonempty output requires explicit `--resume`; otherwise the runner fails rather than overwriting results.

## 10. CLI and scripts

The package exposes:

```text
python -m nrgbd_eval doctor --config configs/h20.json --model MODEL
python -m nrgbd_eval check --config configs/h20.json
python -m nrgbd_eval run --config configs/h20.json --model MODEL --device cuda:0
python -m nrgbd_eval summarize --run-dir ABSOLUTE_PATH
```

Supported model names:

- `vggt`
- `vggt_long`
- `streamvggt`
- `vggt_slam`
- `vggt_omega`

H20 launch scripts accept a physical GPU index, perform identity/disk/GPU/input/model checks, map that physical GPU through `CUDA_VISIBLE_DEVICES`, and run the backend on logical `cuda:0`. They run in the foreground and do not submit or schedule jobs.

## 11. Timing and resource reporting

The package records separately:

- synchronized model-forward time
- adapter preprocessing/postprocessing time
- scorer time
- complete per-scene wall time
- complete run wall time
- PyTorch peak allocated bytes
- PyTorch peak reserved bytes
- periodic `nvidia-smi` used-memory samples
- process peak CPU RSS from `/usr/bin/time -v`

Model-forward timing excludes checkpoint loading, dataset I/O, scoring, and result serialization. Resume summaries keep the maximum peak memory across all invocations and sum only committed scene timings. External GPU samples may include other processes and are labelled accordingly.

## 12. Configuration

`configs/h20.json` contains only paths and protocol/model settings. No credentials are stored. It records:

- dataset root
- output root
- reference source
- five model source roots and checkpoints
- Python interpreter per backend
- `kf=10`
- input and crop resolutions
- point cap
- ICP threshold
- deterministic seed
- resource admission threshold

All paths are direct paths, not symlinks created by the infrastructure.

## 13. Testing

CPU tests use synthetic images, depths, poses, point clouds, and fake adapters. They must not read real checkpoints or allocate CUDA memory.

Required tests cover:

- exact nine-scene selection and exclusion of `archives`
- numeric frame ordering and every-tenth selection
- missing modality and malformed pose rejection
- millimetre-to-metre depth conversion and invalid-depth masking
- OpenGL-to-OpenCV pose conversion
- crop/resize intrinsics consistency
- GT isolation from adapters
- exact prediction frame identity/order validation
- known rigid/scale geometry and ICP metrics
- deterministic point sampling
- empty/degenerate prediction failure
- atomic scene completion
- interrupted-run resume and provenance mismatch refusal
- partial versus complete summary
- allocation-free backend creation and doctor
- resource/timing schema
- all five backend registrations
- CLI JSON output and shell syntax

A source-level contract test forbids metric implementation imports inside adapter modules.

## 14. Acceptance criteria

The infrastructure is ready when:

1. `doctor` describes all five backends without loading a scene.
2. `check` validates all nine real NRGBD scenes without CUDA or checkpoint reads.
3. All CPU unit and integration tests pass.
4. Formatting, linting, JSON parsing, and shell syntax checks pass.
5. No real model evaluation has been started.
6. No data, checkpoint, cache, output, or secret is included in Git.
7. Documentation explains exact commands, protocol, outputs, resume, timing, VRAM, and known unverified items.
8. The source snapshot is published under `VGGTFamily/eval/nrgbd`.

Real GPU compatibility, runtime, memory, and benchmark scores remain unverified until a separately authorized model run.
