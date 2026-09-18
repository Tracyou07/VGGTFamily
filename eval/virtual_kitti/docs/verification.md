# Virtual KITTI 1.3.1 verification gates

**CPU infra verified; formal evaluation blocked by data/protocol.**
CPU acceptance, a real-data smoke and formal-table acceptance are separate gates.
No real dataset is complete here. No GPU/model numerical validation or benchmark
resource measurement was performed.

## Gate 1: independent CPU infrastructure

From this repository root, with the existing H20 environment:

```bash
export PYTHON=/home/ubuntu/anaconda3/envs/fastwam/bin/python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES= \
  "$PYTHON" -B -m pytest -q -p no:cacheprovider tests
bash scripts/verify_all_cpu.sh
git diff --check
```

The suite covers strict data contracts, synthetic metric geometry, finite atomic
JSON, provenance binding, resource-boundary ordering, isolated process failures,
exact resume, aggregate completeness and exact export columns. Standalone-copy
acceptance imports this package with only its own source path and exercises CLI
help while rejecting sibling evaluator imports. Static scans also reject obsolete
evaluator paths. No shared parent runtime or cross-dataset runner is required.

The verification script repeats the complete local suite, help and shell syntax.
On H20/Linux it compares deterministic snapshots from `scripts/snapshot_repository.py`
before and after execution, including regular-file SHA-256, byte size, device/inode,
file type and permissions; directory and symlink identities are recorded too.
Content checks include ignored artifacts and already-dirty tracked files, and do
not require Git metadata. Same-size overwrites with restored modification time
still fail. An unchanged starting edit is allowed; a changed byte or file identity
is not. Git checkouts also check tracked generated artifacts and whitespace.

Only these explicit benign locations are excluded: `.git`, `.superpowers`,
`.pytest_cache`, `__pycache__`, `.mypy_cache`, `.ruff_cache`, Python `.pyc`/`.pyo`
files, and the configured model-owned Long dependency cache. Other repository
files, including results, outputs and prepared artifacts, remain covered.
Symlink targets are recorded without following or hashing target contents.
Directory-relative file descriptors and no-follow opens keep traversal inside the
repository, including when it contains links to external model or dataset trees.
Special filesystem objects are recorded without opening them. File changes during
hashing cause a failure instead of an accepted mixed snapshot.

The script disables bytecode/cache generation and CUDA visibility. Tests create
disposable synthetic data and CPU checkpoint fixtures in temporary directories;
they do not download data, load real models, install dependencies, or write external
model/dataset trees.
Virtual KITTI fixture tests additionally enforce release/layout and image decode
gates, real 1.3.1 extrinsic conventions, condition identity during resume, and
structured failures for corrupt compressed geometry and overflowing alignments.

Initial packaging CPU suite snapshot (2026-09-15): **191 passed** with the existing
H20 Python 3.10.20 environment, CUDA hidden, and no dataset/model installation.
This is fixture evidence only.

## Gate 2: read-only readiness

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" -B -m virtual_kitti_eval doctor --config configs/h20.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src "$PYTHON" -B -m virtual_kitti_eval doctor --config configs/h20.json --model vggt_long
```

See [data-status.md](data-status.md) for the current blockers. Model doctor and
dataset doctor answer different questions. Import/checkpoint readiness is not
model correctness. The local Long dependency cache must be populated through a
later explicitly requested offline setup before Long can be ready.

## Gate 3: deferred real-data smoke

After complete data and an applicable protocol are identified, recheck H20
identity, disk space, active jobs, GPU visibility/memory, intended project,
interpreter, checkpoint and source fingerprints. Prepare and verify one explicit
scene/condition, then run the lightest ready model serially. No current CLI
`--max-frames` flag is promised: any bounded-frame smoke requires an explicitly
defined prepared-data/provenance policy before implementation; never silently
slice formal inputs.

Inspect original frame IDs (without synthetic timestamps), pose direction, camera coordinates,
metric alignment and failure state before accepting numbers. Verify synchronized
inference seconds, peak allocated MiB, peak reserved MiB and actual before/after
GPU snapshots. Repeat the exact command with `--resume` and verify zero additional
backend launches. Keep source and input identities fixed during the comparison.

## Gate 4: deferred formal-table acceptance

Obtain and verify complete official 1.3.1 RGB and extrinsics for Scene01/02 with
all six main conditions. Existing 2.0.3 files cannot satisfy this release gate.
Accept each condition independently, preserve all twelve exact columns, and retain
any explicitly configured extra scenes separately. Do not substitute a condition
or present an average in the tracking table.

Record real-data/GPU evidence in a separate report and commit. Only that evidence
can support numerical-validation or formal-evaluation claims. A successful
explicit subset or supplemental protocol does not satisfy this gate.
