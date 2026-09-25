# v5 development verification

2026-09-21. Base v4 17439bd3ee46c8f2aab681fa0fc3baffdb8f1dd0; branch codex/ours-v5.

| Suite | Result |
| --- | --- |
| New v5 CPU tests | 27/27 passed |
| Existing v4 CPU/infra regression | 24/24 passed |
| Existing results-root regression | 8/8 passed |
| Total | 59/59 passed |
| Python compile of every new production module | Passed |
| Worker/reference/campaign CLI --help | Passed |
| bash -n scripts/run_v5.sh | Passed |
| Real FP32/BF16 GPU gates | NOT RUN |
| 100-frame two-mode diagnostic | NOT RUN |
| 500/1000-frame memory or geometry validation | NOT RUN |

## What the CPU tests establish

- True allowed-edge SDPA agrees with a small dense-mask oracle (float64 atol/rtol 1e-9).
  No global square mask is passed. Camera has local all/remote camera dependencies only;
  noncamera cannot directly read a remote window. Camera changes reach patches through
  the next original frame attention. BF16 CPU scatter uses actual output dtype.
- Tiny real Aggregator + original Block/Attention/RoPE execute frame/global layers.
  Independent mode reproduces original cached features. Window-first reference,
  tail lengths, separate overlap storage, ordering/batch invariance and frozen weights pass.
- Tiny real CameraHead and DPT heads test the complete wrapper. The fixture fixes only
  untrained FoV output rows to a valid range; no production/head weights are changed.
- Same valid input gives identical Sim3 rotation/translation/scale to the frozen original
  Long function. Known-transform, outliers, shuffled IDs, rank/finite/empty failures,
  chain direction, camera rotation/depth-point consistency and ownership pass.
- CPU worker smoke executes actual tiny model, reconstruction, evaluation and PLY/NPZ/PNG
  export in a temporary directory. CUDA APIs/checkpoint/preflight are mocked, so this is
  NOT CUDA/1B-checkpoint evidence. Global files omit stale local pose_encoding.
- Gate acceptance requires exact completed passed report sets, hashes, correct 55-frame
  scope and matching contract. Contract includes source/checkpoint, scene/full fixed list,
  RGB metadata, PyTorch/CUDA and fixed tolerance identity. Busy GPU/low disk rejection
  are CPU mocks, not an actual resource reservation or GPU execution.

## Review

Independent read-only reviewer checked topology, stale-state aliasing, memory lifetimes,
reference isolation and runtime gates. Found and fixed: worker dictionary syntax;
local pose_encoding leaking into global artifacts; incomplete gate report acceptance;
missing dataset/runtime seal identity. Final focused review confirmed fixes.
No unresolved correctness finding was reported. This does not substitute for real GPU gates.

## Preservation and limitations

Original model/layer/head sources and reference snapshot are unchanged from v4. Existing
v4 worktree remained clean. No optimizer/training, downloads, dataset migration, GPU model
execution, old result deletion or overwrite occurred. New files are confined to ours_v5.
Source checkpoints are referenced, not copied; runtime workers rehash before use.

Memory includes every window's GPU token state and head caches, with overlap duplicates.
Allowed-edge execution removes forbidden pair computation but does not make activation
memory constant in sequence length. Long alignment first-use Numba compilation can add
CPU time. GPU tolerance values are predeclared v4 budgets, not measured v5 accuracy.

Non-blocking inherited warnings: original VGGT CUDA autocast API deprecation, and an evo
ResourceWarning in an existing 7Scenes protocol test. Neither altered pass/fail criteria.

Raw test outputs: cpu_v5_tests.txt, cpu_v4_regression.txt, cpu_results_root.txt.
Machine-readable totals and hashes: TEST_RESULTS.json. README contains explicit commands.

Whitespace check: new implementation passes git diff --check. Frozen vendor files retain upstream trailing whitespace and EOF layout deliberately so their byte hashes match; they are excluded from formatting checks.
