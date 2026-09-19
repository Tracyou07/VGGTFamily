import argparse
import json
from pathlib import Path
from .data import preflight_dataset
from .backends import NAMES, create_backend, doctor_backend
from .runner import run
from .results import summarize


def _load(path):
    p = Path(path).resolve()
    d = json.loads(p.read_text())
    d["_config_path"] = str(p)
    return d


def main(argv=None):
    p = argparse.ArgumentParser(prog="nrgbd-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("check", "doctor", "run"):
        q = sub.add_parser(cmd)
        q.add_argument("--config", required=True)
        if cmd != "check":
            q.add_argument("--model", required=True, choices=NAMES)
        if cmd == "run":
            q.add_argument("--device", default="cuda:0")
            q.add_argument("--resume", action="store_true")
            q.add_argument("--scene", action="append")
    q = sub.add_parser("summarize")
    q.add_argument("--run-dir", required=True)
    a = p.parse_args(argv)
    if a.cmd == "summarize":
        from .data import SCENES

        out = summarize(Path(a.run_dir), SCENES)
    else:
        c = _load(a.config)
        if a.cmd == "check":
            out = preflight_dataset(
                c["dataset_root"], int(c.get("protocol", {}).get("kf", 10))
            )
        elif a.cmd == "doctor":
            out = doctor_backend(a.model, c["models"][a.model])
        else:
            out = run(
                c,
                create_backend(a.model, c["models"][a.model], a.device),
                a.resume,
                a.scene,
            )
    print(json.dumps(out, indent=2, sort_keys=True))
    if a.cmd == "doctor" and not out["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
