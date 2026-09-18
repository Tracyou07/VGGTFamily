# 7-Scenes RGB-registered evaluation depth

H20 production dataset: `/data/yjh/share/datasets/7scenes_registered_simplerecon_v1`.
Prepared on 2026-09-14: **17,000 test frames, 18 sequences, 7 scenes**. This directory contains only the test split.

The source `/data/yjh/share/datasets/7scenes` is preserved. Every output `frame-XXXXXX.depth.proj.png` is a generated, regular uint16 PNG. RGB and pose files are symlinks to their original files. The evaluation aliases `/data/sy/7scenes` and `/home/ubuntu/yjh/feedforwardreconstruct/vggtstream/data/eval/7scenes` resolve to the registered dataset.

## Geometry

`registration.py` implements the [SimpleRecon 7-Scenes preprocessing convention](https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py):

- Backproject raw depth using focal length 585 and principal point `(320, 240)`.
- Apply the published depth-to-RGB rigid transform, then project using RGB focal length 525.
- Preserve the reference's `+0.5` source pixel centers, nearest-even pixel rounding, nearest-Z collision handling, and float32 raster conversion to integer millimeters.
- Exclude raw values 0 and 65535 before projection. Unobserved RGB pixels remain zero; holes are not interpolated.

These are published approximate calibration parameters used by the benchmark. They do not establish a new per-device calibration. Renaming or linking raw depth does not perform these transformations.

## Generate or verify

Run on H20, from `/home/ubuntu/yjh/feedforwardreconstruct`, using the existing `vggt-gx` environment. Verification reads all 17,000 projected PNG hashes:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B \
  eval/7scenes/preprocessing/prepare_7scenes.py --verify
```

For a new build, pass a new output directory; the builder refuses existing output directories and never writes into the source tree:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B \
  eval/7scenes/preprocessing/prepare_7scenes.py \
  --source-root /data/yjh/share/datasets/7scenes \
  --output-root /data/yjh/share/datasets/7scenes_registered_new_build --workers 8
```

`registration.json` records calibration, source-code hash, counts, completion state, and the checksum of `frames.jsonl`. The frame manifest records raw-depth and registered-depth hashes. Each generated PNG is decoded again to verify lossless storage before its completion record is written.

## Evaluation contract

The VGGT/VGGT*/FastVGGT evaluator and the StreamVGGT, Omega, and current Long adapters default to registered data. Their preflight validates completion, test splits, RGB/pose presence and every registered-depth hash before model allocation. Missing projected files and per-file symlinks are rejected; there is no raw-depth fallback.

Each evaluation requires an empty output directory and writes `input_registration.json`. The four shell launchers add a timestamp, PID and `registered_v1` to result/log names. Stream broadcasts preflight failures so every distributed worker receives the same error.

Existing runs started before this fix retain their previously loaded reader and raw data root. Their geometric metrics must be recomputed using registered GT. An `input_registration.json` identifies the input protocol; it is not a certificate that all other model-specific processing is correct. Separate audit findings, including Omega's image/GT cropping mismatch, are outside this depth repair.

The historical `/tmp/7scenes_eval_proc` cache is retained and marked invalid for RGB GT. It includes training sequences as well as test sequences; it was not converted or removed. The unpublished first build at `/data/yjh/share/datasets/7scenes_registered_simplerecon_v1_pre_quantization_fix_20260914` is retained with `complete: false` and must not be evaluated.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/anaconda3/envs/vggt-gx/bin/python -B -m unittest discover \
  -s eval/7scenes/preprocessing/tests -v
```

The 19 tests cover geometry, quantization, occlusion, invalid depth, real PNG generation, input corruption, output isolation, distributed error propagation, both actual loaders, and all six current evaluation CLIs. CPU smoke checks also read one real frame from each of the 18 sequences through both loaders and compare depth, intrinsics and valid masks. No GPU inference is required by these checks.

The 6 existing adapter tests also pass (25 automated tests total). A separate two-rank CPU/Gloo smoke check verifies preflight-error delivery to both workers.

H20 evidence is under `/home/ubuntu/yjh/feedforwardreconstruct/eval/7scenes/preprocessing/logs`. Original edited code is backed up under the sibling `backups` directory. `install_registered_inputs.py` records the narrow edits against their original source text and stops if that text has changed; do not rerun it on the already patched checkout.

