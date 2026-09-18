from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

COMMANDS = ("doctor", "prepare", "verify", "run", "aggregate", "export-table")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="virtual-kitti-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        if name in ("doctor", "prepare", "verify"):
            command.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2]/"configs/h20.json")
            command.add_argument("--sequence")
        elif name in ("aggregate", "export-table"):
            command.add_argument("--output", type=Path, required=True)
        elif name == "run":
            command.add_argument("--config",type=Path,required=True)
            command.add_argument("--model",required=True)
            command.add_argument("--sequence",nargs="+",action="extend",required=True)
            command.add_argument("--output",type=Path,required=True)
            command.add_argument("--device",default="cuda:0")
            command.add_argument("--timeout",type=float,default=3600.)
            command.add_argument("--resume",action="store_true")
        if name == "doctor":
            command.add_argument("--model",help="Read-only native model import probe")
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from .runner import preflight_run, run_evaluation
        try:
            plan = preflight_run(args.config,args.model,args.sequence,args.output,args.device,timeout_s=args.timeout)
            summary = run_evaluation(plan,resume=args.resume)
            requested_complete = all(s in summary.per_sequence for s in plan.sequence_ids) and "_manifest" not in summary.failures
            payload = asdict(summary)
            payload.update(requested_complete=requested_complete,formal_complete=summary.complete)
            print(json.dumps(payload,sort_keys=True,allow_nan=False))
            return 0 if requested_complete else 1
        except (OSError,ValueError) as exc:
            print(json.dumps({"requested_complete":False,"formal_complete":False,"code":getattr(exc,"code","RUN_FAILED"),
                              "message":str(exc),"diagnostics":getattr(exc,"diagnostics",{})},allow_nan=False))
            return 1
    from .config import DatasetValidationError, load_config
    from .data import inspect_raw_sequence, prepare_sequence, verify_prepared_sequence
    from .results import aggregate, export_table
    try:
        if args.command in ("aggregate", "export-table"):
            if args.command == "aggregate":
                summary = aggregate(args.output)
                print(json.dumps(asdict(summary), allow_nan=False))
                return 0 if summary.complete else 1
            table = export_table(args.output)
            print(table, end="")
            from .io import read_json
            return 0 if read_json(args.output / "summary.json")["complete"] else 1
        config = load_config(args.config)
        if args.command == "doctor" and args.model:
            from .backends import doctor_backend, normalize_model_key
            try:
                key = normalize_model_key(args.model)
                status = doctor_backend(key,config.models[key])
                print(json.dumps(asdict(status),sort_keys=True,allow_nan=False))
                return 0 if status.ready else 1
            except (KeyError,ValueError) as exc:
                print(json.dumps({"ready":False,"blockers":[{"code":"BACKEND_CONFIG","message":str(exc)}]}))
                return 1
        sequences = (args.sequence,) if args.sequence else config.sequence_ids
        if args.command == "doctor":
            statuses = [inspect_raw_sequence(config, sequence) for sequence in sequences]
            ready = all(status.ready for status in statuses)
            print(json.dumps({"ready": ready, "dataset_version": config.dataset_version,
                "sequences": [{"sequence": s.sequence, "ready": s.ready,
                    "blockers": [asdict(b) for b in s.blockers]} for s in statuses]}, allow_nan=False))
            return 0 if ready else 1
        # Preflight every requested sequence before creating any prepared output.
        statuses = [inspect_raw_sequence(config, sequence) for sequence in sequences]
        failed = next((s for s in statuses if not s.ready), None)
        if failed is not None:
            raise DatasetValidationError(failed.blockers[0].code, failed.blockers[0].message)
        action = prepare_sequence if args.command == "prepare" else verify_prepared_sequence
        prepared = [action(config, sequence) for sequence in sequences]
        print(json.dumps({"ready": True, "sequences": [
            {"sequence": p.sequence, "frames": len(p.frame_ids), "manifest": str(p.manifest_path)}
            for p in prepared]}, allow_nan=False))
        return 0
    except (DatasetValidationError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"ready": False, "code": getattr(exc, "code", "EVALUATION_ERROR"),
            "message": str(exc), "command": args.command}, allow_nan=False))
        return 1
