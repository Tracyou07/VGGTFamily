# Virtual KITTI 1.3.1 model and worker contract

This contract is implemented and tested with CPU fixtures. No real GPU/model
numerical validation has been performed. Native execution remains gated by
[data readiness](data-status.md) and the selected model's read-only doctor.

## Request boundary

The parent verifies prepared input before creating a `BackendRequest`.
The schema-version-1 `worker/request.json` has exactly these fields:

```text
schema_version, model_key, model_config, frame_ids, image_paths, output_dir,
request_id, provenance_id, sequence, device
```

`model_key` is one of `vggt`, `vggt_star`, `streamvggt`, `vggt_slam`,
`vggt_long`, `vggt_omega`. Display aliases normalize at the API/CLI boundary;
FastVGGT is rejected. `model_config` is recursively immutable and contains
absolute `interpreter`, `project_root` and `checkpoint` plus all native
controls. `request_id` distinguishes attempts; `provenance_id` binds run input.
`output_dir` is an absolute attempt directory, and `device` is a logical CUDA
device within the caller's visible selection.

`sequence` is the exact `SceneXX/Condition` selection. `frame_ids` are original
ordered five-digit RGB filename stems and must match image paths. No GT labels
or calibration arrays are sent to the model. `PreparedSequence.timestamps_s` is
`None`; this dataset contract supplies frame indexes rather than measured times.

Only RGB paths and original frame identities cross the dataset/model boundary. Ground-truth poses, LiDAR, depth, semantic/instance labels and
calibration arrays remain in the parent. Native temporary image names may be
contiguous indexes, but an explicit mapping restores original output IDs.
Predictions must retain exactly the request's frame order and count.

## Process and result boundary

The launcher starts the configured interpreter as
`-B -m virtual_kitti_eval.backend_worker --request request.json` in a fresh process group.
It writes exclusive request/log files and rejects reused attempt artifacts.
The child owns caches and native output in the attempt directory. A timeout
terminates the entire child process group.

The child atomically publishes `prediction.npz`, then `worker_result.json`.
A successful worker JSON has exactly:

```text
schema_version, status, failure_code, message, request_id, provenance_id,
model_key, sequence, prediction_sha256, metadata, resources
```

Success requires schema 1, `status="success"`, null `failure_code`, matching
identities, zero process return code, no signal, matching NPZ SHA-256, and finite
nonnegative resources. `resources` contains `inference_seconds`,
`peak_allocated_mib`, `peak_reserved_mib`, `nvidia_smi_before`,
`nvidia_smi_after`. Snapshot dictionaries retain query output or a query error.

NPZ is loaded with `allow_pickle=False` and contains only `frame_ids`
(Unicode vector), `poses_c2w` (N,4,4), and optional `world_points` (M,3).
Metadata must explicitly declare `pose_convention="c2w"` and
`pose_scale="rigid"`. Pose matrices must be finite, proper rigid homogeneous
transforms; reflections, scale embedded in rotation, unknown direction,
missing/reordered IDs and malformed arrays fail validation. Monocular global
scale is aligned later by the metric's positive Sim(3), not hidden in rotations.

Release 1.3.1 extrinsics map world to camera. Preparation inverts the rigid
homogeneous matrix to c2w. The documented camera uses right/down/forward axes.
ATE scores translation centers in metres; no GT camera pose or calibration reaches
native inference. Condition paths retain their exact identity during resume,
including when an alternate path points to a different valid condition.

OOM, timeout, signal, launch errors, missing/corrupt predictions, identity/hash
mismatches and metric errors are terminal structured failures. Failed attempts
never provide an accepted prediction or resource usage to the runner, and never
qualify for resume. The parent writes the strict
[result/metric pair](output-schema.md) only after metric validation.

## Model profile and dependencies

All six adapters use native model implementations at explicitly configured
external source locations. Strict checkpoint key/shape loading happens inside
the child before timing. CPU doctor checks checkpoint containers and required
imports, using disposable temporary caches with CUDA/network/external writes
forbidden. A ready doctor is not evidence that model numerical output is correct.

The selected Virtual KITTI Long profile is chunk 75, overlap 30, loop chunk 20,
SALAD retrieval, loop closure enabled and Sim(3) stitching. This is a project
selection, not a claim of published Virtual KITTI tuning.
SLAM's profile retains submap size 16 and maximum loops 1. Applicable image,
confidence, chunk, loop, retrieval and reconstruction choices are configuration
and provenance inputs. Long's four pinned dependencies are installed in the model-owned runtime
`/home/ubuntu/yjh/feedforwardreconstruct/vggtlong/.runtime/long_deps`. Both dataset evaluators reference this direct path, and
`scripts/setup_long_deps.sh` reproduces the installation from an offline
wheelhouse. Long doctor validates every import without constructing the model.

## Timing boundary

`load()` finishes model and retrieval initialization; then the worker resets
CUDA peak statistics, captures the before snapshot, synchronizes, starts a
monotonic timer, runs `infer()`, synchronizes, and stops timing.
Preprocessing, forward, decoding, native reconstruction, stitching/loop closure,
and output extraction are included. Loading, resource-query latency, serialization
and parent metric evaluation are excluded. Long preload wrappers are idempotent
to prevent native `run()` from hiding a second load inside timing.

Allocated/reserved byte peaks are divided by `2**20`. Peak allocated MiB is the
main table field; peak reserved MiB is retained in JSON. Already resident model
memory contributes to the peak. CPU facade tests validate ordering and units;
they do not measure real GPU timing or memory.
