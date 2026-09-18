# Virtual KITTI 1.3.1 evaluation

**CPU infra verified; formal evaluation blocked by data/protocol.**
This standalone repository owns its data adapter, metrics, six native backends,
isolated worker, runner, provenance, aggregation and export. CPU fixture checks
do not establish real-data compatibility or GPU/model numerical correctness.

| Gate | Current status |
|---|---|
| CPU infrastructure | Verified with synthetic fixtures and isolated subprocesses |
| Real data | Existing 2.0.3 paths and incomplete RGB archive do not satisfy formal 1.3.1 |
| Formal protocol/table | Requires Scene01/02 across six separate conditions |
| Long dependencies | Verified in the model-owned runtime; Long doctor passes |
| Real GPU inference | Not performed; no benchmark scores or measured GPU resources accepted |

See [data status](docs/data-status.md), [verification gates](docs/verification.md),
[model contract](docs/model-contract.md), and [output schema](docs/output-schema.md).
See also [the release-specific data format](docs/data-format.md).

## Install and CPU verification

Run from this repository root. The existing H20 environment needs no installation:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/eval/virtual_kitti
export PYTHON=/home/ubuntu/anaconda3/envs/fastwam/bin/python
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -B -m virtual_kitti_eval --help
bash scripts/verify_all_cpu.sh
```

For a separate environment, the optional install command is
`python -m pip install -e '.[test]'`. The base package requires NumPy and Pillow;
the test extra includes pytest, Torch and safetensors for CPU checkpoint/worker
fixtures. Use an appropriate Torch distribution for that environment. Native
models continue to use their explicitly configured existing interpreters and
source/checkpoint paths. No packages were installed for this verification.

`verify_all_cpu.sh` selects only this repository's `src` and tests, disables
CUDA visibility, bytecode and pytest caches, checks standalone-copy imports and
CLI help, scans for sibling evaluator dependencies and obsolete evaluator paths,
and validates shell syntax. Its local content/identity snapshot detects changed
bytes, replacement files, added/deleted paths and permissions even when ignored
or already-dirty files retain the same Git status. It never follows symlinks into
external models or datasets. Git checkouts additionally reject tracked generated
artifacts and whitespace errors. See the explicit snapshot exclusions in
[verification.md](docs/verification.md). It does not invoke data preparation or a native model.

## Data layout

Formal input is release **1.3.1**, monocular RGB:
`raw_root/vkitti_1.3.1_rgb/<world>/<variation>/%05d.png` and
`raw_root/vkitti_1.3.1_extrinsicsgt/<world>_<variation>.txt`.
Worlds are `0001`, `0002`, `0006`, `0018`, `0020`; canonical
`Scene01/Clone` maps to `0001/clone` at the data boundary. Prepared state is
`prepared_root/SceneXX/Condition/{manifest.json,poses_c2w.npy}`.
The optional `VERSION` marker and all observed release-bearing directories must agree.

Paths resolve relative to the explicit config file. Doctor and verify are read-only.
Prepare publishes validated prepared state and rejects an existing invalid target;
it never downloads or silently replaces data. Edit a config only after identifying
the intended complete dataset location.

```bash
"$PYTHON" -B -m virtual_kitti_eval doctor --config configs/h20.json
"$PYTHON" -B -m virtual_kitti_eval doctor --config configs/h20.json --model vggt_long
bash scripts/preflight_h20.sh
"$PYTHON" -B -m virtual_kitti_eval prepare --config configs/h20.json --sequence Scene01/Clone
"$PYTHON" -B -m virtual_kitti_eval verify --config configs/h20.json --sequence Scene01/Clone
```

The checked-in config deliberately inspects existing 2.0.3 paths while requiring
1.3.1, so doctor reports `DATASET_VERSION_MISMATCH` and
`INCOMPLETE_RGB_ARCHIVE`. Text GT alone is insufficient. Original five-digit RGB
IDs and rigid c2w poses are verified; depth and semantic/instance labels are unused.
No artificial timestamps are introduced. The main list is Scene01/02 crossed with
Clone, Fog, Morning, Overcast, Rain and Sunset.

## Run one model, resume, aggregate and export-table

Only after dataset verification and the selected model's doctor are ready,
run one explicitly selected scene/condition. These examples are future execution commands;
the current data blockers prevent them from producing a formal evaluation.

```bash
CUDA_VISIBLE_DEVICES=7 "$PYTHON" -B -m virtual_kitti_eval run --config configs/h20.json \
  --model vggt --sequence Scene01/Clone --output results/vggt-one --device cuda:0 --timeout 3600
CUDA_VISIBLE_DEVICES=7 "$PYTHON" -B -m virtual_kitti_eval run --config configs/h20.json \
  --model vggt --sequence Scene01/Clone --output results/vggt-one --device cuda:0 --timeout 3600 --resume
"$PYTHON" -B -m virtual_kitti_eval aggregate --output results/vggt-one
"$PYTHON" -B -m virtual_kitti_eval export-table --output results/vggt-one
```

The equivalent wrapper is `bash scripts/run_one.sh vggt Scene01/Clone results/vggt-one`;
append `--resume` for the identical request. The caller controls physical CUDA
visibility; `cuda:0` means the first visible GPU. No command starts a model matrix.
Runs hold an exclusive lock and execute requested scene/conditions serially.
Resume accepts only successful result/metrics pairs matching current frame IDs,
config, sources, checkpoints, interpreter, auxiliary assets and run provenance.
Stale or partial attempts are quarantined under `failures/` before rerun.

Run exits 0 only if every requested scene/condition succeeds; `requested_complete`
and `formal_complete` are separate. Aggregate and export write summaries in the
selected existing output directory and exit 1 for an incomplete formal run.
Unavailable values are `—`, and failures never enter formal averages.
Export prints Markdown and never edits a paper table.

Tracking columns are exactly `Model | Calibration | Scene 01 Clone | Scene 01 Fog |
Scene 01 Morning | Scene 01 Overcast | Scene 01 Rain | Scene 01 Sunset |
Scene 02 Clone | Scene 02 Fog | Scene 02 Morning | Scene 02 Overcast |
Scene 02 Rain | Scene 02 Sunset | Status`. Every condition is independent;
there is no tracking average column. Resource columns are `Model | Scene / condition |
Input frames | Inference time (s) ↓ | Peak VRAM (MiB) ↓ | Status`.
Explicitly configured Scene06/18/20 pairs remain separate machine-readable records.

## Models and model-owned Long setup

Supported keys are `vggt`, `vggt_star`, `streamvggt`, `vggt_slam`,
`vggt_long`, and `vggt_omega`. FastVGGT is unsupported.
Every source, checkpoint and interpreter is explicit in `configs/h20.json`.
Model doctor validates paths/checkpoint containers and probes imports without
model construction, CUDA initialization, network access or external writes.
Dependency readiness is distinct from dataset readiness and numerical validation.

Long's dependency path resolves to the model-owned runtime
`/home/ubuntu/yjh/feedforwardreconstruct/vggtlong/.runtime/long_deps`. KITTI and Virtual KITTI use the same pinned model dependency
tree through this direct path; no symlink and no Conda mutation is involved.
The current doctor passes. Recreating it offline uses:

```bash
PYTHON=/path/to/configured/model/python bash scripts/setup_long_deps.sh /path/to/wheelhouse
```

The setup command installs only `faiss-cpu==1.8.0.post1`, `llvmlite==0.44.0`,
`numba==0.61.2`, and `pypose==0.9.5` into this local target with
`--no-index --no-deps`. It performs no downloads or model-environment installation.
The wheelhouse must already contain wheels compatible with the configured interpreter.
The selected Virtual KITTI Long profile is chunk 75, overlap 30, loop chunk 20,
SALAD retrieval, loop closure enabled and Sim(3) stitching. This is a project
selection, not a claim of published Virtual KITTI tuning.

## Timing boundary and VRAM

Model and retrieval loading finish before CUDA peak reset and the synchronized
timer. The measured interval includes RGB preprocessing, forward passes, camera
decoding, native reconstruction, output extraction, stitching and loop closure.
Model loading, external `nvidia-smi` query latency, serialization and parent metric
computation are outside inference time. The worker synchronizes before starting
and after inference, and retains before/after resource snapshots.

The resource table's main VRAM value is **peak allocated MiB**.
**Peak reserved MiB remains in JSON**. Both divide CUDA bytes by `2**20`;
the peaks include already resident model allocations plus inference allocations.
These are defined contracts, not measured real-GPU results from this task.

## GPU admission policy

The H20 config requires at least 81920 MiB free and no active compute processes on the selected physical GPU. This is a conservative scheduling threshold, not a model-memory estimate. Users may deliberately lower gpu_min_free_mib in a copied config; gpu_require_no_compute_processes controls the idle requirement. The read-only check resolves CUDA_VISIBLE_DEVICES indices or UUID prefixes, queries memory and compute processes during preflight and immediately before every worker, and fails closed with GPU_QUERY_FAILED, GPU_MEMORY_INSUFFICIENT, or GPU_BUSY and structured diagnostics. It never signals other processes. Admission is a snapshot, not an exclusive reservation; other launchers can still race the check.
