# Virtual KITTI data and model readiness

**CPU infra verified; formal evaluation blocked by data/protocol.**
Read-only H20 doctor check: 2026-09-16, using the checked-in config.

- `DATASET_VERSION_MISMATCH`: formal release is 1.3.1, but existing configured
  paths under `/data/yjh/share/datasets/Virtual_KITTI_2.0.3` identify 2.0.3.
- `INCOMPLETE_RGB_ARCHIVE`: the existing RGB archive is partial.
- The checked paths have text GT, which cannot replace complete RGB and extrinsics
  for the required 1.3.1 release. Dataset doctor exits 1.
- No main Scene01/02 × Clone/Fog/Morning/Overcast/Rain/Sunset condition is accepted
  as a complete real-data evaluation input.
- Long model doctor exits 0. `pypose`, `numba`, `llvmlite` and `faiss`
  import from `/home/ubuntu/yjh/feedforwardreconstruct/vggtlong/.runtime/long_deps`.


The config intentionally reports the mismatch instead of relabeling 2.0.3 data.
Dataset status is dynamic; rerun doctor before preparing or evaluating.
A ready software probe for a model would not satisfy this release/data gate.

No Virtual KITTI data conversion, real preparation or GPU inference was performed.
The pinned Long dependencies were installed outside the Conda environment. See the
[release-specific format](data-format.md), [README commands](../README.md)
and [deferred verification gates](verification.md).
