# Repository Guidelines

## Project Structure & Module Organization

`vggt/` is the Python package. `vggt/models/` contains the top-level `VGGT`
model and aggregator, `vggt/heads/` contains prediction heads, `vggt/layers/`
contains Transformer building blocks, and `vggt/utils/` contains geometry,
pose, loading, and visualization helpers. Third-party-derived tracking code is
kept under `vggt/dependency/`. Keep the `main` worktree limited to reusable
baseline code; research programs, data lists, and generated results belong on
their dedicated experiment branches.

## Build, Test, and Development Commands

- `pip install -r requirements.txt` installs core inference dependencies.
- `pip install -e .` installs the package in editable mode for development.
- `python -c "from vggt.models.vggt import VGGT; print(VGGT.__name__)"` checks
  that the package imports successfully.
- `python -m unittest discover -s tests` runs tests when a `tests/` directory is
  present. New tests should remain CPU-only unless CUDA is intrinsic to the case.

## Coding Style & Naming Conventions

Target Python 3.10+ and follow the surrounding PyTorch style. Use four spaces
for indentation, `snake_case` for functions and variables, and `CamelCase` for
classes. Group imports as standard library, third-party packages, then local
modules. Document non-obvious tensor shapes and coordinate conventions at API
boundaries. No repository-wide formatter is configured, so avoid unrelated
formatting or refactoring.

## Testing Guidelines

Use `unittest` for focused regression tests. Name files `test_<subject>.py` and
test methods `test_<behavior>`. Cover tensor shapes, argument forwarding,
default compatibility, and failure cases before relying on checkpoint-level
experiments. Tests must not download model weights or private datasets.

## Commit & Pull Request Guidelines

History uses short imperative subjects such as `Add camera iteration probe`.
Keep commits scoped to one purpose. Pull requests should state the motivation,
changed interfaces, verification commands, and any required runtime resources.
Never commit checkpoints, licensed datasets, access tokens, or machine-specific
paths.

## Worktree Policy

A worktree is an additional local checkout, not a replacement for a branch.
Keep every active experiment worktree attached to its own named branch; do not
use detached HEAD for continuing research. Use `phenomenon-characterization`
for observation studies and `camera-iteration-preexperiment` for method
pre-experiments. Commit research artifacts on the corresponding branch rather
than adding them to `main`.

Remote machines cannot fetch local worktree metadata. Reproduce on AutoDL from
a pushed branch or recorded commit. Keep `VGGT_DiT_Research_Guide.md` and
`VGGT_DiT_Implementation_Plan.md` only on `main`; worktrees may keep only
clearly named, branch-specific designs and logs.
