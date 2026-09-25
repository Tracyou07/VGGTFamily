# ours_v4 validation report

Campaign: `/data/yjh/output/vggt/ours_v4/20260921_equivalence_ccebb0b`

Implementation tested: `ccebb0b61ee8d886e836e8ea45499b92ec7a45cb`. CPU tests: 16/16 passed.
Original VGGT*: `cc1d8ac15861aea54d14961653cd340e7d984f29`. Separate original reference package, unmodified model attention.
Stitching source: ours_v3 `d54c6e2c6e8e0491d04bd7d4105e455217aeb884`.
Checkpoint SHA256: `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
Same fixed scene0150_00 frame IDs and preprocessed tensors; no GT or alignment in equivalence comparisons.

## Equivalence gates

55 frames: [0,30), [20,50), [40,55); packed groups [[0,1],[2]]. Full encoder and head-cache tensors were compared, not samples.

- fp32 repeat: **PASS**; fixed tolerances `{'atol': 0.0005, 'rtol': 0.0005, 'center_m': 0.0001, 'rotation_deg': 0.02}`.
  - window_0000.pt: center 0 m; rotation 2.6998693e-06 deg; failed tensor fields: [].
  - window_0001.pt: center 0 m; rotation 2.9575587e-06 deg; failed tensor fields: [].
  - window_0002.pt: center 0 m; rotation 2.6998693e-06 deg; failed tensor fields: [].
- fp32 equivalence: **PASS**; fixed tolerances `{'atol': 0.0005, 'rtol': 0.0005, 'center_m': 0.0001, 'rotation_deg': 0.02}`.
  - window_0000.pt: center 3.1828894e-08 m; rotation 3.6222548e-06 deg; failed tensor fields: [].
  - window_0001.pt: center 4.4046395e-08 m; rotation 4.9783131e-06 deg; failed tensor fields: [].
  - window_0002.pt: center 0 m; rotation 2.6998693e-06 deg; failed tensor fields: [].
- bf16 repeat: **PASS**; fixed tolerances `{'atol': 0.02, 'rtol': 0.02, 'center_m': 0.01, 'rotation_deg': 0.5}`.
  - window_0000.pt: center 0 m; rotation 3.1945285e-06 deg; failed tensor fields: [].
  - window_0001.pt: center 0 m; rotation 2.9575587e-06 deg; failed tensor fields: [].
  - window_0002.pt: center 0 m; rotation 2.4148365e-06 deg; failed tensor fields: [].
- bf16 equivalence: **PASS**; fixed tolerances `{'atol': 0.02, 'rtol': 0.02, 'center_m': 0.01, 'rotation_deg': 0.5}`.
  - window_0000.pt: center 3.0264401e-08 m; rotation 3.1945285e-06 deg; failed tensor fields: [].
  - window_0001.pt: center 3.3788532e-08 m; rotation 4.9783131e-06 deg; failed tensor fields: [].
  - window_0002.pt: center 0 m; rotation 2.4148365e-06 deg; failed tensor fields: [].

## Resources

| Run | Forward seconds | Allocated GiB | Reserved GiB | CPU peak GiB |
|---|---:|---:|---:|---:|
| bf16_packed | 10.350 | 39.818 | 63.193 | 10.214 |
| bf16_reference | 16.917 | 40.461 | 74.137 | 10.217 |
| bf16_repeat | 16.752 | 40.461 | 74.137 | 10.214 |
| fp32_packed | 28.877 | 39.819 | 63.047 | 10.214 |
| fp32_reference | 35.838 | 40.460 | 41.180 | 10.212 |
| fp32_repeat | 35.930 | 40.460 | 41.180 | 10.217 |
| packed100 | 15.611 | 39.961 | 63.193 | 10.212 |
| reference100 | 28.733 | 40.341 | 74.137 | 10.212 |

Gate resource measurements include feature-capture hooks. They are not a substitute for the 100-frame timing comparison. Peak reserved memory can increase with batching.

## 100-frame comparison

- reference100: ATE 0.04073979 m (one global Sim3 for evaluation only).
- packed100: ATE 0.04073979 m (one global Sim3 for evaluation only).
- Raw per-window numerical differences: `100_frame_raw_window_differences.json`; no Sim3 applied to this comparison.

## Historical v3 context

Existing v3 100-frame results (20260920T112529Z): local_shared_ref ATE 0.10254751 m; camera_global 0.10510145 m. v3 uses a different shared-state backbone, so comparisons cannot be interpreted as a batching-only effect.

## Limitations

Only the fixed 30/10 window campaign is validated here. No 1000-frame run, no feature cache, no training. Full FP32/BF16 validity requires both gates to pass. Failed tensor equivalence cannot be overridden by a visually plausible point cloud or a lower ATE.

Commands and architecture: `README_OURS_V4.md`. Gate configuration: `configs/v4_validation.json`. Existing versions and outputs were left untouched.

## 100-frame stitching details

| Path | Forward s | Stitch s | End-to-end s | Boundary translation error m (mean) | Boundary rotation error deg (mean) |
|---|---:|---:|---:|---:|---:|
| reference100 | 28.7333 | 1.0987 | 62.8543 | 0.05111966 | 1.43677332 |
| packed100 | 15.6113 | 1.1473 | 49.8034 | 0.05111965 | 1.43676494 |

End-to-end includes checkpoint loading/hash, inference, serialization, stitching and evaluation. Single paired run, not a statistically established speedup.

| Path / edge | Scale B to A | Inliers | Ratio | Inlier RMSE (local prediction units) |
|---|---:|---:|---:|---:|
| reference100/edge_0000_0001 | 1.135745926 | 9864 | 0.979835 | 0.015278147 |
| reference100/edge_0001_0002 | 0.946510274 | 9351 | 0.896462 | 0.020560430 |
| reference100/edge_0002_0003 | 0.977845764 | 15691 | 0.784550 | 0.022919585 |
| reference100/edge_0003_0004 | 1.073647777 | 18013 | 0.900650 | 0.023353421 |
| packed100/edge_0000_0001 | 1.135745951 | 9864 | 0.979835 | 0.015278147 |
| packed100/edge_0001_0002 | 0.946510261 | 9351 | 0.896462 | 0.020560427 |
| packed100/edge_0002_0003 | 0.977845774 | 15691 | 0.784550 | 0.022919584 |
| packed100/edge_0003_0004 | 1.073647778 | 18013 | 0.900650 | 0.023353421 |

Independent post-run checks: 100 unique frame IDs, finite poses, proper rotations, ATE recomputed from saved evaluation arrays. Both original repositories ours_v2 and ours_v3 remained clean.

H20 preflight: VM-0-11-ubuntu / ubuntu; GPU 1 UUID GPU-bbfe5861-f0ef-1a6b-dfb3-32ac3e5281d4 was idle (4 MiB allocated, 0% utilization). /data had 43 GiB available. Existing GPU jobs on other devices were preserved. No training, 1000-frame run or batch benchmark started.
