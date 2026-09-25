# VGGTFamily evaluation infrastructure and v10 source

Independent evaluation infrastructure for VGGT-family reconstruction models. Each dataset has its own directory and instructions.

| Dataset | Entry point | Instructions |
| --- | --- | --- |
| 7-Scenes | `eval/7scenes/run_*.sh` and adapters | [README](eval/7scenes/README.md) |
| ScanNet | `python -m scannet_eval` | [README](eval/scannet/README.md) |
| KITTI | `python -m kitti_eval` | [README](eval/kitti/README.md) |
| Virtual KITTI | `python -m virtual_kitti_eval` | [README](eval/virtual_kitti/README.md) |
| NRGBD | `python -m nrgbd_eval` | [README](eval/nrgbd/README.md) |

## Getting started

Clone this repository, choose a dataset, and follow its README. Install that dataset's package and model dependencies in the appropriate Python environment. The v10 model source snapshot is in [ours_v10](ours_v10/README_v10.md). Checkpoints and datasets remain external inputs.

The `configs/h20.json` files and shell launchers preserve the current H20 deployment settings. Before using another machine, replace the absolute dataset, checkpoint, model repository, Python environment, and output paths. The published repository layout is not automatically substituted for `/home/ubuntu/yjh/feedforwardreconstruct`. Use direct paths for datasets. Do not treat unregistered raw depth as RGB-aligned ground truth.

For ScanNet, KITTI, Virtual KITTI, and NRGBD, consult their CLI help and README for input checks, model diagnostics, execution, and resume. Each dataset owns its selection rules, metrics, provenance, timing, and resource reporting. GPU runs require the corresponding model environment and available hardware.

## Scope of this snapshot

The evaluation infrastructure was snapshotted from H20 on 2026-09-19; the v10 source snapshot was added on 2026-09-25. It includes current code, configurations, preprocessing, documentation, and CPU tests where available. It does not contain datasets, pretrained weights, caches, run logs, experiment results, archives, or Waymo. The `ours_v10/` directory is a source-only snapshot of the v10 implementation; separate earlier `ours` worktrees and runtime artifacts are not bundled, although the v10 tree retains shared historical modules. No new model evaluation was started for this publication. Historical verification documents describe the checks performed when those components were developed. The v10 snapshot retains H20-specific paths in its scripts and needs configuration for another machine.

7-Scenes is currently a deployment-oriented script collection; the other four datasets have separate Python packages. This snapshot does not claim that every backend has been verified on a clean external machine.

## v10 source provenance

The source-only v10 snapshot exports 290 tracked files from commit `73dd68890eb4584486c820aa4594ae79b34bee51` on branch `codex/ours-v10`. See [source snapshot details](ours_v10/SOURCE_SNAPSHOT.md) and [v10 usage notes](ours_v10/README_v10.md).

## Third-party code

Protocol reference code is retained with the licenses shipped by its authors. See [third-party notices](THIRD_PARTY_NOTICES.md). Model repositories and checkpoints retain their own license and usage conditions. No blanket license is asserted over the entire snapshot.
