# Virtual KITTI 1.3.1 output schema

All JSON is finite and duplicate-key rejecting. NaN, Infinity and overflowing
JSON exponents are rejected; writes encode with `allow_nan=False`, flush/fsync a sibling
temporary file and atomically replace the destination. A result is accepted only
through its strict validator; filenames alone do not establish success.

## Layout

```text
output/
  run_manifest.json
  SceneXX/Condition/metrics.json
  SceneXX/Condition/result.json
  SceneXX/Condition/worker/request.json
  SceneXX/Condition/worker/prediction.npz
  SceneXX/Condition/worker/worker_result.json
  SceneXX/Condition/worker/stdout.log
  SceneXX/Condition/worker/stderr.log
  SceneXX/Condition/failure.json       # detailed terminal failure, when present
  all_sequences_metrics.json
  summary.json
  failures/                    # structured failures and quarantined attempts
```

A live run also holds `.run.lock`. Stale locks require inspection; a runner never
silently steals one. Historical failure files remain, and current `summary.json`
is authoritative. Worker wire fields and coordinate conventions are documented
in [model-contract.md](model-contract.md).

## Result commit marker

`result.json` has exactly these schema-version-1 fields:

| Field | Type / meaning |
|---|---|
| `schema_version` | Integer 1 |
| `model_key` | Canonical model string |
| `sequence` | Exact canonical scene/condition path |
| `status` | `success`, `oom`, `error`, or `timeout` |
| `input_frames` | Positive integer |
| `inference_seconds` | Finite nonnegative seconds; null allowed on failure |
| `peak_allocated_mib` | Finite nonnegative MiB; null allowed on failure |
| `peak_reserved_mib` | Finite nonnegative MiB; null allowed on failure |
| `worker_exit_state` | Exactly `{returncode: integer-or-null, signal: positive-integer-or-null}` |
| `provenance_id` | Run's content-bound SHA-256 identity |
| `metrics_sha256` | SHA-256 of canonical metric JSON; null on failure |

Success requires `returncode=0`, null signal, nonnull resources and matching
metrics. `write_result_pair` validates the full pair, writes metrics first, then
the result as the hash-bound commit marker. Failures have no accepted metric hash.
An old metric file cannot make a failed or interrupted result resumable.
Inference seconds exclude parent metric computation. The main resource table
uses peak allocated MiB; reserved MiB remains machine-readable JSON.

## Metric schema

`metrics.json` has exactly `schema_version=1`, `protocol_id`, `matched_frames`,
`rmse_m`, `alignment`, `frame_ids_sha256`, `sequence`, `model_key`, `provenance_id`.
The protocol is `virtual-kitti-1.3.1-ate-sim3-v1`; `rmse_m` is finite nonnegative
camera-center ATE RMSE in metres. Original ordered five-digit IDs are hashed.
No timestamp or tracking-average field is fabricated.

Alignment contains exactly `scale` (positive), `rotation` (proper 3x3),
and `translation` (length 3). The translation is in metres and scale is
dimensionless. ATE compares camera centers after positive proper Umeyama Sim(3).
At least three non-collinear matched centers are required; no RPE or scale-error
column is inferred. Frame inventory and metric counts must match provenance.

## Run provenance

The strict `run_manifest.json` top-level fields are:

```text
schema_version, sequence_file, config_file, config_payload, model_key,
model_config, sequence_ids, sequences, output_dir, device, timeout_s, command,
metric_protocol_id, model_source, checkpoint, interpreter, package_source,
auxiliary_sources, provenance_id
```

File records bind absolute paths and SHA-256. Source records additionally bind
excluded paths; interpreter records include executable path, version and binary
hash. The config, original selection file, exact prepared manifests and frame
inventory, evaluator/native sources, checkpoint, consumed auxiliary assets,
device, output location, timeout and command participate in provenance.
`provenance_id` hashes canonical content excluding itself.
Each `sequences` record includes the exact scene/condition identity, ordered
frame IDs, their hash and prepared-manifest identity. Runner model controls retain
CUDA execution-environment choices. Resume cannot substitute another condition
through an alternate filesystem path.

Source fingerprints include relevant source/config/native-library files and
matching untracked files, while excluding Git metadata, caches, build artifacts,
planning files and explicitly excluded data/output paths. Consumed Long/SALAD/DINO
and SLAM hub inputs are bound separately. Live source and prepared-input checks
are required before accepting cached results. Changing the requested subset or
any bound input changes run identity.

## Aggregation and table readiness

Formal completeness requires all twelve main Scene01/02 ×
Clone/Fog/Morning/Overcast/Rain/Sunset pairs. Every condition has its own metric
column and no tracking average is calculated, so no `average_metrics.json` is
emitted. Explicit optional scene/condition pairs remain separate in machine-readable
output and never create new main-table columns.

OOM/error/timeout, invalid, partial, missing and stale pairs stay visible as
failures and never enter formal averages. `all_sequences_metrics.json` preserves
accepted per-unit metric records; `summary.json` records current completion,
per-unit values, failure maps and resources. Export rebuilds aggregation and
preserves the exact documented columns, using `—` for unavailable values.
