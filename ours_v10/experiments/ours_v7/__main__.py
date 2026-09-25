"""One-mode, arbitrary-length H20 launcher for independent v7 run IDs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.ours_v6.runtime import (ROOT, data_identity, fresh_directory, prepare_inputs,
                      preflight, sha256, source_identity, write_json)
from vggt.v7.attention import MODES
from experiments.ours_v6.windows import make_windows


def validate_reference(config, frame_ids, preprocessing, frame_list):
    path = Path(config["v5_reference_manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"verified v5 reference manifest missing: {path}")
    old = json.loads(path.read_text())
    if old["checkpoint_sha256"] != config["checkpoint_sha256"]:
        raise ValueError("v5 checkpoint identity differs")
    canonical = ROOT / "configs/scene0150_00_frames100.json"
    if Path(frame_list).resolve() == canonical.resolve():
        provenance = path.parent.parent / "provenance.json"
        if not provenance.is_file():
            raise FileNotFoundError("v5 verified provenance file missing")
        recorded_hash = json.loads(provenance.read_text())["frame_list_sha256"]
        if recorded_hash != sha256(frame_list):
            raise ValueError("v5 fixed frame-list hash differs")
        if old["frame_ids"][:len(frame_ids)] != list(frame_ids):
            raise ValueError("v5 fixed frame IDs/order differ")
        for key in ("loader", "dtype", "minimum", "maximum"):
            if old["preprocessing"][key] != preprocessing[key]:
                raise ValueError(f"v5 image preprocessing differs: {key}")
        if old["preprocessing"]["shape"][1:] != preprocessing["shape"][1:]:
            raise ValueError("v5 image preprocessing shape differs")
    return dict(manifest=str(path), v5_commit=old["sources"]["commit"],
                checkpoint_sha256=old["checkpoint_sha256"],
                frame_prefix_verified=Path(frame_list).resolve() == canonical.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--frame-list", type=Path)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--patch-exchange-ratio", type=float)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/v7_validation.json").read_text())
    output_root = Path("/data/yjh/output/vggt/ours_v7").resolve()
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(output_root):
        parser.error("--output must be a new path inside /data/yjh/output/vggt/ours_v7")
    if not args.gpu.isdigit():
        parser.error("--gpu must be a physical GPU index")
    if args.frames < 1:
        parser.error("--frames must be positive")
    window_size = int(config["window_size"]) if args.window_size is None else args.window_size
    overlap = config["overlap"] if args.overlap is None else args.overlap
    ratio = float(config["patch_exchange_ratio"]) if args.patch_exchange_ratio is None else args.patch_exchange_ratio
    if not 0 <= ratio <= 1:
        parser.error("--patch-exchange-ratio must lie in [0,1]")
    windows = make_windows(args.frames, window_size, overlap)
    scene = args.scene_root or Path(config["scene_root"])
    frame_list = args.frame_list or ROOT / "configs/scene0150_00_frames100.json"
    identity = source_identity()
    if not identity["commit"] or identity["status"]:
        raise RuntimeError("ours_v7 requires a committed clean tree")
    if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint identity changed")
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    lock = open(f"/tmp/ours_v6_gpu_{args.gpu}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sampler = None
    sample_log = None
    try:
        information = preflight(args.gpu, args.output)
        fresh_directory(args.output)
        try:
            write_json(args.output / "preflight.json", information)
            saved = prepare_inputs(scene, frame_list, args.output / "inputs.pt", args.frames)
            verified = validate_reference(config, saved["frame_ids"], saved["preprocessing"], frame_list)
            contract = dict(code=identity, data=data_identity(scene, frame_list),
                            checkpoint_sha256=config["checkpoint_sha256"],
                            reference_v5=verified, scene=str(scene), frame_list=str(frame_list),
                            frame_ids=saved["frame_ids"], windows=windows,
                            mode=args.mode, window_size=window_size, overlap=overlap,
                            patch_exchange_ratio=ratio,
                            patch_selection_algorithm=config["patch_selection_algorithm"],
                            precision="bf16", preprocessing=saved["preprocessing"],
                            window_batching="disabled; every window processed separately")
            write_json(args.output / "config.json", dict(base=config, contract=contract))
            del saved
            sample_log = (args.output / "vram.csv").open("x")
            sampler = subprocess.Popen(
                ["nvidia-smi", "-i", args.gpu,
                 "--query-gpu=timestamp,uuid,memory.used,memory.free,utilization.gpu",
                 "--format=csv", "--loop-ms=500"],
                stdout=sample_log, stderr=subprocess.STDOUT
            )
            command = [sys.executable, "-B", "-u", "-m", "experiments.ours_v7.worker",
                       "--input", str(args.output / "inputs.pt"),
                       "--output", str(args.output / args.mode),
                       "--gpu", args.gpu, "--frames", str(args.frames),
                       "--mode", args.mode, "--window-size", str(window_size),
                       "--overlap", str(overlap), "--patch-exchange-ratio", str(ratio)]
            with (args.output / f"{args.mode}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = args.output / args.mode
            if not (result / "COMPLETE.json").is_file():
                raise RuntimeError("worker did not complete")
            worker_manifest = json.loads((result / "run_manifest.json").read_text())
            contract["patch_selection"] = worker_manifest["patch_selection"]
            write_json(args.output / "config.json", dict(base=config, contract=contract))
            write_json(args.output / "summary.json",
                       json.loads((result / "evaluation_summary.json").read_text()))
            write_json(args.output / "COMPLETE.json",
                       dict(status="complete", mode=args.mode, frames=args.frames,
                            windows=windows, worker=str(result)))
        except Exception:
            write_json(args.output / "FAILED.json", dict(traceback=traceback.format_exc()))
            raise
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=10)
        if sample_log is not None:
            sample_log.close()
        lock.close()


if __name__ == "__main__":
    main()
