from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

COMMANDS = ("doctor", "prepare", "verify", "run", "aggregate", "export-table")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kitti-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command_parser = sub.add_parser(name)
        if name in ("doctor", "prepare", "verify"):
            command_parser.add_argument("--config", type=Path, required=True)
            command_parser.add_argument("--sequence", nargs="+", action="extend",
                                        help="Explicit sequence subset; default is the configured list")
        if name == "doctor":
            command_parser.add_argument("--model", help="Inspect one native model in its configured interpreter")
        if name in ("aggregate", "export-table"):
            command_parser.add_argument("--output", type=Path, required=True)
        if name == "run":
            command_parser.add_argument("--config", type=Path, required=True)
            command_parser.add_argument("--model", required=True)
            command_parser.add_argument("--sequence", nargs="+", action="extend", required=True)
            command_parser.add_argument("--output", type=Path, required=True)
            command_parser.add_argument("--device", default="cuda:0")
            command_parser.add_argument("--timeout", type=float, default=3600.)
            command_parser.add_argument("--resume", action="store_true")
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("aggregate", "export-table"):
        from .results import aggregate, export_table, read_json
        try:
            if args.command == "aggregate":
                summary = aggregate(args.output)
                print(json.dumps(asdict(summary), sort_keys=True, allow_nan=False))
                return 0 if summary.complete else 1
            print(export_table(args.output), end="")
            return 0 if read_json(args.output / "summary.json")["complete"] else 1
        except (OSError, ValueError) as exc:
            print(json.dumps({"complete": False, "code": "INVALID_OUTPUT", "message": str(exc)}))
            return 1
    if args.command == "run":
        from .runner import preflight_run, run_evaluation
        try:
            plan = preflight_run(args.config, args.model, args.sequence, args.output,
                                 args.device, timeout_s=args.timeout)
            summary = run_evaluation(plan, resume=args.resume)
            requested_complete = all(s in summary.per_sequence for s in plan.sequence_ids)
            requested_complete = requested_complete and "_manifest" not in summary.failures
            payload = asdict(summary)
            payload.update(requested_complete=requested_complete, formal_complete=summary.complete)
            print(json.dumps(payload, sort_keys=True, allow_nan=False))
            return 0 if requested_complete else 1
        except (OSError, ValueError) as exc:
            print(json.dumps({"requested_complete": False, "formal_complete": False,
                              "code": getattr(exc, "code", "RUN_FAILED"), "message": str(exc),
                              "diagnostics": getattr(exc, "diagnostics", {})}, allow_nan=False))
            return 1
    from .config import DatasetValidationError, load_config
    from .data import inspect_raw_sequence, prepare_sequence, verify_prepared_sequence
    try:
        config = load_config(args.config)
    except DatasetValidationError as exc:
        print(json.dumps({"ready": False, "blockers": [{"code": exc.code, "message": exc.message}]}))
        return 1
    if args.command == "doctor" and args.model:
        from .backends import doctor_backend, normalize_model_key
        try:
            key = normalize_model_key(args.model)
            status = doctor_backend(key, config.models[key])
            print(json.dumps(asdict(status), sort_keys=True, allow_nan=False))
            return 0 if status.ready else 1
        except (KeyError, ValueError) as exc:
            print(json.dumps({"ready": False, "blockers": [{"code": "BACKEND_CONFIG", "message": str(exc)}]}))
            return 1
    sequence_ids = tuple(dict.fromkeys(args.sequence or config.sequence_ids))
    results, blockers = {}, []
    for sequence in sequence_ids:
        if args.command == "doctor":
            status = inspect_raw_sequence(config, sequence)
            result = {"ready": status.ready, "blockers": [asdict(b) for b in status.blockers]}
            if status.ready and (config.prepared_root / sequence / "manifest.json").exists():
                try:
                    verify_prepared_sequence(config, sequence)
                except DatasetValidationError as exc:
                    result = {"ready": False, "blockers": [{"code": exc.code, "message": exc.message}]}
        else:
            try:
                prepared = (prepare_sequence if args.command == "prepare" else verify_prepared_sequence)(config, sequence)
                result = {"ready": True, "blockers": [], "manifest_path": str(prepared.manifest_path),
                          "manifest_sha256": prepared.manifest_sha256, "frame_count": len(prepared.frame_ids)}
            except DatasetValidationError as exc:
                result = {"ready": False, "blockers": [{"code": exc.code, "message": exc.message}]}
        results[sequence] = result
        blockers.extend({"sequence": sequence, **b} for b in result["blockers"])
    ready = all(result["ready"] for result in results.values())
    print(json.dumps({"ready": ready, "blockers": blockers,
                      "paths": {k: str(getattr(config, k)) for k in ("raw_root", "color_root", "aux_root", "archive_root", "prepared_root", "sequences_file")},
                      "sequences": results}, sort_keys=True))
    return 0 if ready else 1
