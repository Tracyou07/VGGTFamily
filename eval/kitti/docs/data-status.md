# KITTI data and model readiness

**Raw data and VGGT-Long dependencies verified on H20.**
Read-only/full-decode check: 2026-09-16, using the checked-in config.

- `color_root` directly reads the complete color extraction.
- `aux_root` directly reads calibration and official poses.
- Dataset doctor exits 0: sequences 00–10 are all ready with no blockers.
- Sequence 00 real prepare and verify exit 0 for 4,541 frames.
- Long model doctor exits 0. `pypose`, `numba`, `llvmlite` and `faiss`
  import from `/home/ubuntu/yjh/feedforwardreconstruct/vggtlong/.runtime/long_deps`.
- No raw-data files were copied, moved or linked.

Dataset paths and readiness are dynamic; rerun `doctor --config configs/h20.json`
before evaluation. GPU admission and real model inference remain separate gates.
