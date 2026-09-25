# v5 implementation plan

User-approved specification: camera queries read local all + remote cameras; other queries local only. Independent overlap instances, 30/10, GPU-resident state, original heads and frozen weights. Long point-head overlap IRLS, no loop optimization. CPU development only this run.

Base: 17439bd3ee46c8f2aab681fa0fc3baffdb8f1dd0. Independent clone branch codex/ours-v5; v4 unchanged.

- [x] T1: CPU tests for topology/dense reference, dtype, camera relay and ordering; implement vggt/v5/attention.py.
- [x] T2: CPU tiny-real-aggregator tests for independent equivalence, reference/tail/cache identity, group invariance; implement scheduler.py and model wrapper with original heads.
- [x] T3: same-input original Long oracle tests, transform/degen/ownership; implement frozen vendor alignment adapter and stitcher.
- [x] T4: CLI/preflight/manifests/failure tests; implement diagnostic runner and predeclared GPU gates.
- [x] T5: original v4 tests, review, document commands/limits, commit.

Decisions: do not change original vggt model files or reference/. New wrapper owns the synchronization. Camera K/V can be projected from camera input alone because original norm/QKV/QK-norm/RoPE act per token; all bank entries are immutable old-state snapshots. This avoids retaining all-window full QKV. Original heads remain per-window (same-length groups allowed). Float64 CPU tolerance 1e-9; FP32/BF16 GPU budgets fixed in config before running. No GPU verification claimed. Invalid geometry is rejected, never silently filtered or rescued; valid-input Long numeric path unchanged.

Review fixes: worker syntax now compilation-tested; global artifacts omit stale local pose_encoding; gate seal requires all expected passed hashed reports and 55-frame coverage; seal binds scene/frame list/RGB metadata and torch/CUDA versions. Independent reviewer verified fixes with CPU mocks; no GPU initialized.

Implementation skill: executing-plans, inline implementation. User expressly authorized implementation after reviewing the full spec, so no additional approval round was needed.
