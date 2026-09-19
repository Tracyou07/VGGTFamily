# VGGTFamily evaluation infrastructure

Independent evaluation infrastructure for VGGT-family reconstruction models. Each dataset has its own directory and instructions.

| Dataset | Entry point | Instructions |
| --- | --- | --- |
| 7-Scenes | `eval/7scenes/run_*.sh` and adapters | [README](eval/7scenes/README.md) |
| ScanNet | `python -m scannet_eval` | [README](eval/scannet/README.md) |
| KITTI | `python -m kitti_eval` | [README](eval/kitti/README.md) |
| Virtual KITTI | `python -m virtual_kitti_eval` | [README](eval/virtual_kitti/README.md) |
| NRGBD | `python -m nrgbd_eval` | [README](eval/nrgbd/README.md) |

## Getting started

Clone this repository, choose a dataset, and follow its README. Install that dataset's package and model dependencies in the appropriate Python environment. Model source repositories, checkpoints, and datasets are external inputs and are not included.

The `configs/h20.json` files and shell launchers preserve the current H20 deployment settings. Before using another machine, replace the absolute dataset, checkpoint, model repository, Python environment, and output paths. The published repository layout is not automatically substituted for `/home/ubuntu/yjh/feedforwardreconstruct`. Use direct paths for datasets. Do not treat unregistered raw depth as RGB-aligned ground truth.

For ScanNet, KITTI, Virtual KITTI, and NRGBD, consult their CLI help and README for input checks, model diagnostics, execution, and resume. Each dataset owns its selection rules, metrics, provenance, timing, and resource reporting. GPU runs require the corresponding model environment and available hardware.

## Scope of this snapshot

This is an infrastructure source snapshot from H20, updated on 2026-09-19. It includes current code, configurations, preprocessing, documentation, and CPU tests where available. It does not contain datasets, pretrained weights, caches, run logs, experiment results, archives, Waymo, or the private `ours` model implementation. No real model evaluation was started for publication. Historical verification documents describe the checks performed when those components were developed.

7-Scenes is currently a deployment-oriented script collection; the other four datasets have separate Python packages. This snapshot does not claim that every backend has been verified on a clean external machine.

## Third-party code

Protocol reference code is retained with the licenses shipped by its authors. See [third-party notices](THIRD_PARTY_NOTICES.md). Model repositories and checkpoints retain their own license and usage conditions. No blanket license is asserted over the entire snapshot.
