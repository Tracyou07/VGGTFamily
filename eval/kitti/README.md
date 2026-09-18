# KITTI Odometry evaluation

**CPU infrastructure and raw KITTI 00–10 are verified; formal evaluation still requires prepared sequences and GPU runs.**
This standalone repository owns its data adapter, metrics, six native backends,
isolated worker, runner, provenance, aggregation and export. CPU fixture checks
do not establish real-data compatibility or GPU/model numerical correctness.

| Gate | Current status |
|---|---|
| CPU infrastructure | Verified with synthetic fixtures and isolated subprocesses |
| Real data | Verified: all canonical sequences 00–10 pass full decode/hash doctor |
| Formal protocol/table | Requires all 11 canonical sequences 00–10 |
| Long dependencies | Verified in the model-owned runtime; Long doctor passes |
| Real GPU inference | Not performed; no benchmark scores or measured GPU resources accepted |

See [data status](docs/data-status.md), [verification gates](docs/verification.md),
[model contract](docs/model-contract.md), and [output schema](docs/output-schema.md).


## Install and CPU verification

Run from this repository root. The existing H20 environment needs no installation:

```bash
cd /home/ubuntu/yjh/feedforwardreconstruct/eval/kitti
export PYTHON=/home/ubuntu/anaconda3/envs/fastwam/bin/python
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -B -m kitti_eval --help
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

The adapter supports a single canonical root and explicit split roots. On H20,
`color_root/sequences/SS/image_2/*.png` and `times.txt` are read directly
from the color extraction, while `aux_root/sequences/SS/calib.txt` and
`aux_root/poses/SS.txt` are read directly from the auxiliary extraction.
Both roots must remain under `raw_root`; no symlink or image copy is used.
The canonical list is 00–10. Preparation retains six-digit
original frame IDs, camera-2 c2w poses, timestamps and calibration. Verification
rehashes raw and prepared files and fully decodes RGB.

Paths resolve relative to the explicit config file. Doctor and verify are read-only.
Prepare publishes validated prepared state and rejects an existing invalid target;
it never downloads or silently replaces data. Edit a config only after identifying
the intended complete dataset location.

```bash
"$PYTHON" -B -m kitti_eval doctor --config configs/h20.json
"$PYTHON" -B -m kitti_eval doctor --config configs/h20.json --model vggt_long
bash scripts/preflight_h20.sh
"$PYTHON" -B -m kitti_eval prepare --config configs/h20.json --sequence 00
"$PYTHON" -B -m kitti_eval verify --config configs/h20.json --sequence 00
```

The current H20 source passes full decode and source hashing for all sequences
00–10. Sequence 00 has also completed real prepare and verify. Model readiness
is checked independently from dataset readiness.

## Run one model, resume, aggregate and export-table

Only after dataset verification and the selected model's doctor are ready,
run one explicitly selected sequence. These examples are future execution commands;
the current data blockers prevent them from producing a formal evaluation.

```bash
CUDA_VISIBLE_DEVICES=7 "$PYTHON" -B -m kitti_eval run --config configs/h20.json \
  --model vggt --sequence 00 --output results/vggt-one --device cuda:0 --timeout 3600
CUDA_VISIBLE_DEVICES=7 "$PYTHON" -B -m kitti_eval run --config configs/h20.json \
  --model vggt --sequence 00 --output results/vggt-one --device cuda:0 --timeout 3600 --resume
"$PYTHON" -B -m kitti_eval aggregate --output results/vggt-one
"$PYTHON" -B -m kitti_eval export-table --output results/vggt-one
```

The equivalent wrapper is `bash scripts/run_one.sh vggt 00 results/vggt-one`;
append `--resume` for the identical request. The caller controls physical CUDA
visibility; `cuda:0` means the first visible GPU. No command starts a model matrix.
Runs hold an exclusive lock and execute requested sequences serially.
Resume accepts only successful result/metrics pairs matching current frame IDs,
config, sources, checkpoints, interpreter, auxiliary assets and run provenance.
Stale or partial attempts are quarantined under `failures/` before rerun.

Run exits 0 only if every requested sequence succeeds; `requested_complete`
and `formal_complete` are separate. Aggregate and export write summaries in the
selected existing output directory and exit 1 for an incomplete formal run.
Unavailable values are `—`, and failures never enter formal averages.
Export prints Markdown and never edits a paper table.

Tracking columns are exactly `Model | LC | Calibration | Recon. | Avg. | Avg.* |
00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | Status`.
`Avg.` requires all 11 sequence RMSE values; `Avg.*` requires the ten excluding
01. Each is an arithmetic mean in metres. Resource columns are
`Model | Sequence | Frames | Time (s) | Peak VRAM (MiB) | Status`.

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
The KITTI Long profile is chunk 75, overlap 30, loop chunk 20, SALAD retrieval,
loop closure enabled and Sim(3) stitching.

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

Prepared KITTI sequences are rechecked against current raw readiness before inference and resume acceptance. A new .part archive, changed image order/count, mismatched pose/time counts, or any source-inventory/hash drift blocks execution. Doctor also compares any existing prepared manifest against the current raw inventory. This read-only check includes image decoding and hashing; prepared manifests remain immutable.
