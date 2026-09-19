# NRGBD Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a standalone FastVGGT-compatible NRGBD evaluation infrastructure for five VGGT-family models without starting a real GPU evaluation.

**Architecture:** A shared loader, preprocessing pipeline, scorer, result writer, and runner own the protocol. Five allocation-free backend adapters normalize model-specific outputs into one prediction contract; real model code is imported only inside prediction workers.

**Tech Stack:** Python 3.10, NumPy, SciPy, Pillow, OpenCV/Open3D when scoring, PyTorch only inside real backends, unittest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-19-nrgbd-infra-design.md`

## Global Constraints

- Strict protocol ID `fastvggt_nrgbd_kf10_v1`.
- Exactly nine expected scenes; exclude `archives`.
- `kf=10`, input `(518,392)`, center `224x224`, point cap 999999, ICP threshold 0.1 m.
- Adapters never receive GT depth, points, or poses.
- Do not load real checkpoints, allocate CUDA, or start real evaluation during implementation.
- Use direct paths; publish no datasets, weights, outputs, caches, logs, or secrets.

## Review Focus

- Missing RGB/depth/pose IDs must fail rather than silently change selected frames.
- Pose convention and crop-adjusted intrinsics must match the reference.
- Resume must reject stale protocol, input, source, or checkpoint provenance.
- Backends must remain allocation-free until `predict`.
- Partial summaries must never look canonical.

---

### Task 1: Data and protocol contract

**Files:**
- Create: `pyproject.toml`, `src/nrgbd_eval/contracts.py`, `src/nrgbd_eval/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `load_scene(root, scene_id, protocol) -> SceneInput`, `preflight_dataset(root, protocol) -> dict`
- Consumes: direct NRGBD layout only.

- [ ] Write failing tests for exact scene list, archive exclusion, numeric ordering, every-tenth selection, missing pairs, pose parsing, depth units, and GT-free model inputs.
- [ ] Run `python -m unittest tests.test_data -v`; expect import failures.
- [ ] Implement contracts and data loader.
- [ ] Run the test; expect pass.
- [ ] Commit.

### Task 2: Shared scoring and durable results

**Files:**
- Create: `src/nrgbd_eval/geometry.py`, `src/nrgbd_eval/scoring.py`, `src/nrgbd_eval/results.py`, `src/nrgbd_eval/runner.py`
- Test: `tests/test_scoring.py`, `tests/test_runner.py`

**Interfaces:**
- Consumes: `SceneInput`, `ScenePrediction`.
- Produces: `score_scene`, atomic scene commits, strict resume, complete/partial summaries.

- [ ] Write failing tests for known metrics, deterministic cap, bad clouds, atomic completion, resume fingerprints, and partial summaries.
- [ ] Run focused tests; expect failures.
- [ ] Implement minimal shared scorer and runner.
- [ ] Run focused tests; expect pass.
- [ ] Commit.

### Task 3: Five model adapters

**Files:**
- Create: `src/nrgbd_eval/backends/common.py`, `src/nrgbd_eval/backends/runtime.py`, `src/nrgbd_eval/backends/{vggt,vggt_long,streamvggt,vggt_slam,vggt_omega}.py`
- Test: `tests/test_backends.py`

**Interfaces:**
- Consumes: RGB-only `ModelSceneInput`.
- Produces: allocation-free `create_backend`, nonloading `doctor_backend`, validated `ScenePrediction`.

- [ ] Write failing registration, doctor, allocation, GT-isolation, source/checkpoint validation, and worker-command tests.
- [ ] Run focused tests; expect failures.
- [ ] Implement adapters as strict model-runtime workers with model-specific source/checkpoint/environment definitions and normalized artifact ingestion.
- [ ] Run focused tests; expect pass.
- [ ] Commit.

### Task 4: CLI, configuration, scripts, documentation

**Files:**
- Create: `src/nrgbd_eval/cli.py`, `src/nrgbd_eval/__main__.py`, `configs/h20.json`, `scripts/*.sh`, `README.md`, `.gitignore`, `THIRD_PARTY_NOTICES.md`
- Test: `tests/test_cli.py`, `tests/test_contract.py`

**Interfaces:**
- Produces: `doctor`, `check`, `run`, `summarize`; H20 launchers and resource logs.
- Consumes: interfaces from Tasks 1-3.

- [ ] Write failing CLI JSON, config, adapter/scorer separation, and shell contract tests.
- [ ] Run focused tests; expect failures.
- [ ] Implement CLI, config, launch scripts, reference provenance, and docs.
- [ ] Run focused tests and `bash -n`; expect pass.
- [ ] Commit.

### Task 5: Real-input preflight, full verification, and publication

**Files:**
- Create: `docs/nrgbd-preflight.json`, `docs/verification.txt`
- Modify: GitHub `README.md`, `PUBLICATION_CHECKS.md`; add `eval/nrgbd`.

**Interfaces:**
- Consumes: completed package and H20 NRGBD dataset.
- Produces: verified source snapshot in `Tracyou07/VGGTFamily`.

- [ ] Run full CPU suite, syntax/lint, JSON, shell, secret, and artifact scans.
- [ ] Run `check` against all nine real scenes and save its JSON report.
- [ ] Review the full implementation against spec and fix Important findings with RED→GREEN tests.
- [ ] Copy source-only tree to a fresh clone, verify diff, commit, and push.
- [ ] Confirm local and GitHub commit SHA match.
