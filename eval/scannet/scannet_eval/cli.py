"""Command line interface for ScanNet preparation, checks, and evaluation."""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "h20.json"


def _strict_json(value):
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _config(path):
    from .runner import load_config

    return load_config(path)[1]


def _scenes(values, config):
    from .data import read_scene_list

    if values is None:
        return read_scene_list(config["scene_list"])
    if len(values) == 1 and Path(values[0]).is_file():
        return read_scene_list(values[0])
    flattened = []
    for value in values:
        flattened.extend(part for part in value.split(",") if part)
    if not flattened:
        raise ValueError("at least one scene id is required")
    if len(set(flattened)) != len(flattened):
        raise ValueError("duplicate scene id requested")
    return flattened


def _add_config(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def _add_scenes(parser):
    parser.add_argument("--scenes", nargs="+", metavar="SCENE_OR_FILE")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="scannet-eval", description="FastVGGT-compatible ScanNet evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="prepare immutable ScanNet inputs")
    _add_config(prepare)
    _add_scenes(prepare)
    prepare.add_argument("--raw-root", type=Path)
    prepare.add_argument("--prepared-root", type=Path)
    prepare.add_argument("--max-frames", type=int)
    prepare.add_argument("--workers", type=int, default=4)
    verify = sub.add_parser("verify", help="verify prepared ScanNet inputs")
    _add_config(verify)
    _add_scenes(verify)
    verify.add_argument("--prepared-root", type=Path)
    verify.add_argument("--no-verify-hashes", action="store_true")
    doctor = sub.add_parser(
        "doctor", help="check backend configuration without GPU allocation"
    )
    _add_config(doctor)
    doctor.add_argument("--model", action="append")
    run = sub.add_parser("run", help="evaluate requested scenes")
    _add_config(run)
    _add_scenes(run)
    run.add_argument("--model", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--prepared-root", type=Path)
    run.add_argument("--max-frames", type=int, default=1000)
    run.add_argument("--device", default="cuda")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--gt-ply-dir", type=Path)
    run.add_argument("--chamfer-max-dist", type=float, default=0.5)
    run.add_argument("--plot", action="store_true")
    run.add_argument(
        "--verify-input-hashes",
        action="store_true",
        help="audit every prepared input file hash before inference (default: off)",
    )
    run.add_argument("--backend-config-json", default="{}", help=argparse.SUPPRESS)
    aggregate = sub.add_parser("aggregate", help="rebuild strict aggregate artifacts")
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def _dispatch(args, plan, argv):
    configured = Path(plan.model_config["python"]).resolve()
    current = Path(sys.executable).resolve()
    if configured == current or os.environ.get("SCANNET_EVAL_DISPATCHED") == "1":
        return None
    env = os.environ.copy()
    env["SCANNET_EVAL_DISPATCHED"] = "1"
    repository = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(repository) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [str(configured), "-m", "scannet_eval", *argv], cwd=repository, env=env
    )
    return completed.returncode


def _doctor_one(config_path, config, name):
    model_config = config["models"].get(name)
    if isinstance(model_config, dict) and isinstance(model_config.get("python"), str):
        configured = Path(model_config["python"]).expanduser().resolve()
        if (
            configured != Path(sys.executable).resolve()
            and os.environ.get("SCANNET_EVAL_DOCTOR_WORKER") != name
        ):
            if not configured.is_file():
                return {
                    "ready": False,
                    "errors": [f"configured interpreter does not exist: {configured}"],
                    "diagnostics": {},
                }
            env = os.environ.copy()
            env["SCANNET_EVAL_DOCTOR_WORKER"] = name
            repository = Path(__file__).resolve().parents[1]
            env["PYTHONPATH"] = str(repository) + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            child = subprocess.run(
                [
                    str(configured),
                    "-m",
                    "scannet_eval",
                    "doctor",
                    "--config",
                    str(config_path),
                    "--model",
                    name,
                ],
                cwd=repository,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                return json.loads(child.stdout)["models"][name]
            except (json.JSONDecodeError, KeyError, TypeError):
                return {
                    "ready": False,
                    "errors": [
                        f"doctor subprocess failed ({child.returncode}): {child.stderr.strip()}"
                    ],
                    "diagnostics": {"python": str(configured)},
                }
    from .backends import doctor_backend

    return doctor_backend(name, model_config or {})


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw)
    try:
        if args.command == "prepare":
            config = (
                _config(args.config)
                if args.raw_root is None
                or args.prepared_root is None
                or args.scenes is None
                else {}
            )
            ids = _scenes(args.scenes, config)
            from .data import prepare_dataset

            result = prepare_dataset(
                args.raw_root or config["raw_root"],
                args.prepared_root or config["prepared_root"],
                ids,
                max_frames=args.max_frames,
                workers=args.workers,
            )
            print(_strict_json(result), end="")
            return 0
        if args.command == "verify":
            config = (
                _config(args.config)
                if args.prepared_root is None or args.scenes is None
                else {}
            )
            ids = _scenes(args.scenes, config)
            from .data import validate_dataset

            result = validate_dataset(
                args.prepared_root or config["prepared_root"],
                ids,
                verify_hashes=not args.no_verify_hashes,
            )
            print(_strict_json(result), end="")
            return 0
        if args.command == "doctor":
            config = _config(args.config)
            models = args.model or list(config["models"])
            results = {name: _doctor_one(args.config, config, name) for name in models}
            payload = {
                "ready": bool(results)
                and all(item["ready"] for item in results.values()),
                "models": results,
            }
            print(_strict_json(payload), end="")
            return 0 if payload["ready"] else 1
        if args.command == "aggregate":
            from .runner import aggregate_run

            result = aggregate_run(args.output)
            print(_strict_json(result), end="")
            return 0 if result["complete"] else 1
        config = _config(args.config)
        ids = _scenes(args.scenes, config)
        try:
            overrides = json.loads(args.backend_config_json)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid backend override JSON: {error}") from error
        if not isinstance(overrides, dict):
            raise ValueError("backend override JSON must be an object")
        from .runner import preflight_inputs, run_evaluation

        plan = preflight_inputs(
            args.config,
            args.model,
            args.output,
            ids,
            max_frames=args.max_frames,
            device=args.device,
            prepared_root=args.prepared_root,
            gt_ply_dir=args.gt_ply_dir,
            backend_overrides=overrides,
            chamfer_max_dist=args.chamfer_max_dist,
            verify_input_hashes=args.verify_input_hashes,
        )
        dispatched = _dispatch(args, plan, raw)
        if dispatched is not None:
            return dispatched
        result = run_evaluation(
            args.config,
            args.model,
            args.output,
            ids,
            max_frames=args.max_frames,
            device=args.device,
            resume=args.resume,
            prepared_root=args.prepared_root,
            gt_ply_dir=args.gt_ply_dir,
            backend_overrides=overrides,
            chamfer_max_dist=args.chamfer_max_dist,
            plot=args.plot,
            verify_input_hashes=args.verify_input_hashes,
        )
        print(_strict_json(result), end="")
        return 0 if result["complete"] else 1
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
