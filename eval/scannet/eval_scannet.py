"""Familiar FastVGGT ScanNet entry point backed by strict orchestration."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from scannet_eval.cli import DEFAULT_CONFIG, main as cli_main


def _bool(value):
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parser():
    result = argparse.ArgumentParser(
        description="Evaluate FastVGGT using the preserved ScanNet protocol"
    )
    result.add_argument("--data_dir", type=Path, default=None)
    result.add_argument("--gt_ply_dir", type=Path, default=None)
    result.add_argument("--output_path", type=Path, default=Path("./eval_results"))
    result.add_argument("--input_frame", type=int, default=1000)
    result.add_argument("--ckpt_path", type=Path, default=None)
    result.add_argument("--merging", type=int, default=None)
    result.add_argument("--merge_ratio", type=float, default=0.9)
    result.add_argument("--depth_conf_thresh", type=float, default=1.0)
    result.add_argument("--chamfer_max_dist", type=float, default=0.5)
    result.add_argument("--plot", type=_bool, default=False)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--scenes", nargs="+")
    result.add_argument("--device", default="cuda")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--verify-input-hashes", action="store_true")
    result.add_argument("--vis_attn_map", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.vis_attn_map:
        argument_parser.error(
            "--vis_attn_map is unsupported by the validated backend contract"
        )
    overrides = {
        "depth_conf_thresh": args.depth_conf_thresh,
        "merge_ratio": args.merge_ratio,
    }
    if args.ckpt_path is not None:
        overrides["checkpoint"] = str(args.ckpt_path.expanduser().resolve())
    if args.merging is not None:
        overrides["merging"] = args.merging
    run_root = (
        args.output_path.expanduser().resolve() / f"input_frame_{args.input_frame}"
    )
    command = [
        "run",
        "--config",
        str(args.config),
        "--model",
        "fastvggt",
        "--output",
        str(run_root),
        "--max-frames",
        str(args.input_frame),
        "--device",
        args.device,
        "--chamfer-max-dist",
        str(args.chamfer_max_dist),
        "--backend-config-json",
        json.dumps(overrides, sort_keys=True),
    ]
    if args.data_dir is not None:
        command.extend(["--prepared-root", str(args.data_dir)])
    if args.gt_ply_dir is not None:
        command.extend(["--gt-ply-dir", str(args.gt_ply_dir)])
    if args.scenes:
        command.extend(["--scenes", *args.scenes])
    if args.resume:
        command.append("--resume")
    if args.plot:
        command.append("--plot")
    if args.verify_input_hashes:
        command.append("--verify-input-hashes")
    return cli_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
