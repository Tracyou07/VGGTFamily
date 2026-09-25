"""Read-only CPU verification of three separate v6 GPU run directories."""
import argparse
import json
from pathlib import Path
import numpy as np

MODES = ("independent", "camera_exchange", "camera_register_exchange")
FIELDS = ("c2w", "depth", "intrinsics", "confidence", "world_points", "world_points_conf")


def check(runs, kind):
    if set(runs) != set(MODES):
        raise ValueError("all three modes required")
    contracts = {}
    predictions = {}
    for mode in MODES:
        root = Path(runs[mode])
        if not (root / "COMPLETE.json").is_file():
            raise ValueError(f"incomplete run: {mode}")
        config = json.loads((root / "config.json").read_text())["contract"]
        if config["mode"] != mode:
            raise ValueError("mode/output mismatch")
        worker = root / mode
        if not (worker / "COMPLETE.json").is_file():
            raise ValueError(f"incomplete worker: {mode}")
        with np.load(worker / "global_trajectory.npz", allow_pickle=False) as output:
            ids = output["frame_ids"].astype(str).tolist()
            owners = output["source_window"].astype(int).tolist()
            poses = output["c2w"]
        if ids != config["frame_ids"] or len(ids) != len(set(ids)) or not np.isfinite(poses).all():
            raise ValueError("missing/duplicated/nonfinite final trajectory")
        expected_owner = []
        seen = set()
        for index, (lo, hi) in enumerate(config["windows"]):
            for frame in ids[lo:hi]:
                if frame not in seen:
                    expected_owner.append(index)
                    seen.add(frame)
        if owners != expected_owner:
            raise ValueError("not first-window ownership")
        local = []
        for index, (lo, hi) in enumerate(config["windows"]):
            with np.load(worker / f"windows/{index:04d}/local.npz", allow_pickle=False) as data:
                if data["frame_ids"].astype(str).tolist() != ids[lo:hi]:
                    raise ValueError("window frame mapping changed")
                local.append({key: data[key] for key in FIELDS})
        if not (worker / "evaluation_summary.json").is_file():
            raise ValueError("missing common evaluation summary")
        contracts[mode] = config
        predictions[mode] = local
    common = contracts[MODES[0]]
    for mode in MODES[1:]:
        other = contracts[mode]
        for key in ("frame_ids", "windows", "checkpoint_sha256", "precision"):
            if other[key] != common[key]:
                raise ValueError(f"non-matching comparison input: {key}")
        for key in ("shape", "dtype", "loader", "minimum", "maximum"):
            if other["preprocessing"][key] != common["preprocessing"][key]:
                raise ValueError(f"non-matching preprocessing: {key}")
        if other["data"]["frame_list_sha256"] != common["data"]["frame_list_sha256"]:
            raise ValueError("frame list identity differs")
    if kind == "single":
        if len(common["windows"]) != 1:
            raise ValueError("single-window gate needs one window")
        baseline = predictions["independent"][0]
        for mode in MODES[1:]:
            for key in FIELDS:
                first = baseline[key]
                second = predictions[mode][0][key]
                if not np.array_equal(first, second):
                    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
                    raise ValueError(f"single-window {mode} differs in {key}: max={difference.max()}")
    elif kind == "multi":
        if len(common["windows"]) < 2:
            raise ValueError("multi-window gate needs multiple windows")
    else:
        raise ValueError("unknown gate kind")
    differences = {}
    baseline = predictions["independent"]
    for mode in MODES[1:]:
        window_rows = []
        for index, (reference, compared) in enumerate(zip(baseline, predictions[mode])):
            fields = {}
            for key in FIELDS:
                if reference[key].shape != compared[key].shape:
                    raise ValueError(f"local output shape differs: {mode}/{index}/{key}")
                error = np.abs(reference[key].astype(np.float64) - compared[key].astype(np.float64))
                fields[key] = dict(max_abs=float(error.max()), mean_abs=float(error.mean()))
            window_rows.append(dict(window=index, fields=fields))
        differences[mode] = window_rows
    return dict(kind=kind, frames=len(common["frame_ids"]), windows=common["windows"],
                modes=list(MODES), input_consistent=True, first_window_ownership=True,
                single_window_exact=(kind == "single"),
                local_differences_vs_independent=differences,
                note="Direct local numerical differences include BF16 attention-call-shape effects; they are not all geometry or communication gains.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["single", "multi"], required=True)
    for mode in MODES:
        parser.add_argument("--" + mode.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    runs = {mode: getattr(args, mode) for mode in MODES}
    print(json.dumps(check(runs, args.kind), indent=2))


if __name__ == "__main__":
    main()
