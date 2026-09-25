"""Fixed 100-frame ScanNet comparison: native VGGT-Long and unchanged ours_v5."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from experiments.ours_v5.runtime import ROOT, data_identity, preflight, prepare_inputs, sha256, source_identity, write_json
from experiments.compare_long.evaluate import evaluate_trajectory, write_comparison

SCENE = Path("/data/yjh/share/datasets/ScanNet/prepared_scannet50_v1/scene0150_00")
FRAMES = ROOT / "configs/scene0150_00_frames100.json"
CHECKPOINT = Path("/data/yjh/share/pretrained/VGGT-1B/model.safetensors")
EXPECTED_WINDOWS = [(0, 60), (30, 90), (60, 100)]


def _diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("window prediction shape or finiteness differs")
    error = np.abs(a - b)
    return dict(max_abs=float(error.max()), mean_abs=float(error.mean()), shape=list(a.shape))


def compare_raw_windows(root):
    long_dir = root / "vggt_long/native_long/_tmp_results_unaligned"
    rows = []
    for index, (lo, hi) in enumerate(EXPECTED_WINDOWS):
        native = np.load(long_dir / f"chunk_{index}.npy", allow_pickle=True).item() # freshly produced native file
        mine = {}
        for mode in ("independent", "camera_exchange"):
            with np.load(root / f"ours_v5_{mode}/windows/{index:04d}/local.npz", allow_pickle=False) as data:
                mine[mode] = {k: data[k] for k in ("c2w", "intrinsics", "depth", "world_points", "world_points_conf", "depth_conf")}
            if len(mine[mode]["c2w"]) != hi - lo:
                raise ValueError("ours window length mismatch")
        pairs = (("long_vs_independent", native, mine["independent"]),
                 ("long_vs_exchange", native, mine["camera_exchange"]),
                 ("independent_vs_exchange", mine["independent"], mine["camera_exchange"]))
        for name, left, right in pairs:
            mapping = (("c2w", "extrinsic", "c2w"), ("intrinsics", "intrinsic", "intrinsics"),
                       ("depth", "depth", "depth"), ("world_points", "world_points", "world_points"),
                       ("world_points_conf", "world_points_conf", "world_points_conf"),
                       ("depth_conf", "depth_conf", "depth_conf"))
            difference = {}
            for label, left_key, right_key in mapping:
                first = np.asarray(left[left_key if name.startswith("long_") else right_key]).reshape(hi - lo, -1)
                second = np.asarray(right[right_key]).reshape(hi - lo, -1)
                difference[label] = _diff(first, second)
            rows.append(dict(window=index, start=lo, end=hi, pair=name, fields=difference,
                             note="Direct local-coordinate numeric comparison; no GT or per-window alignment."))
    write_json(root / "raw_window_prediction_differences.json", rows)
    return rows


def _launch(command, cwd, log):
    with Path(log).open("x") as stream:
        subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=["bf16"], required=True)
    parser.add_argument("--frames", type=int, choices=[100], required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or not str(args.output).startswith("/data/yjh/output/vggt/long_compare/"):
        raise ValueError("output must be a new /data/yjh/output/vggt/long_compare/<run_id> directory")
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((ROOT / "configs/v5_validation.json").read_text())
    if (config["window_size"], config["overlap"], config["window_batch_size"]) != (60, 30, 2):
        raise ValueError("ours_v5 fixed 60/30/batch2 config changed")
    if config["checkpoint_sha256"] != sha256(CHECKPOINT):
        raise ValueError("checkpoint hash changed")
    identity = source_identity()
    if not identity["commit"] or identity["status"]:
        raise RuntimeError("comparison requires a committed, clean working tree")
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    lock = open(f"/tmp/ours_v5_gpu_{args.gpu}.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pre = preflight(args.gpu, args.output, min_disk_gib=20)
    args.output.mkdir(parents=True, exist_ok=False)
    sampler = None
    sampler_stream = None
    try:
        write_json(args.output / "preflight.json", pre)
        shared = prepare_inputs(SCENE, FRAMES, args.output / "inputs.pt", 100)
        if tuple(shared["images"].shape) != (100, 3, 392, 518):
            raise ValueError("shared original VGGT preprocessing is not 392x518")
        provenance = dict(scene=str(SCENE), frame_list=str(FRAMES),
                          frame_list_sha256=sha256(FRAMES), frame_ids=shared["frame_ids"],
                          prepared_input_sha256=sha256(args.output / "inputs.pt"),
                          preprocessing=shared["preprocessing"], checkpoint=str(CHECKPOINT),
                          checkpoint_sha256=sha256(CHECKPOINT), ours_commit=identity["commit"],
                          data=data_identity(SCENE, FRAMES), gpu_uuid=pre["gpu_uuid"],
                          precision="bf16", windows=EXPECTED_WINDOWS,
                          long_config_path="/home/ubuntu/yjh/feedforwardreconstruct/eval/scannet/configs/h20.json")
        write_json(args.output / "provenance.json", provenance)
        del shared
        sampler_stream = (args.output / "vram.csv").open("x")
        sampler = subprocess.Popen(["nvidia-smi", "-i", args.gpu,
                                    "--query-gpu=timestamp,uuid,memory.used,memory.free,utilization.gpu",
                                    "--format=csv", "--loop-ms=500"], stdout=sampler_stream,
                                   stderr=subprocess.STDOUT)
        jobs = [("vggt_long", "experiments.compare_long.long_worker"),
                ("ours_v5_independent", "experiments.compare_long.ours_worker"),
                ("ours_v5_camera_exchange", "experiments.compare_long.ours_worker")]
        for name, module in jobs:
            out = args.output / name
            cmd = [sys.executable, "-B", "-u", "-m", module, "--input", str(args.output / "inputs.pt"),
                   "--output", str(out), "--gpu", args.gpu]
            if name != "vggt_long":
                mode = name.removeprefix("ours_v5_")
                cmd += ["--frames", "100", "--mode", mode, "--batch-size", "2",
                        "--window-size", "60", "--overlap", "30"]
            print("START", name, flush=True)
            _launch(cmd, ROOT, args.output / f"{name}.log")
            if not (out / "COMPLETE.json").is_file():
                raise RuntimeError(f"{name} did not complete")
        rows = []
        for name, _ in jobs:
            directory = args.output / name
            metrics, evaluation_timing = evaluate_trajectory(directory / "global_trajectory.npz", SCENE, directory / "common_eval")
            if name == "vggt_long":
                timing = json.loads((directory / "timing.json").read_text())
                native = json.loads((directory / "long_manifest.json").read_text())
                window_config = json.dumps(dict(chunks=native["chunk_indices"],
                                                 overlap=native["resolved_native_config"]["Model"]["overlap"],
                                                 loop_enable=native["resolved_native_config"]["Model"]["loop_enable"]))
            else:
                manifest = json.loads((directory / "run_manifest.json").read_text())
                timing = manifest["timing"]
                model_load = json.loads((directory / "comparison_timing.json").read_text())
                timing["model_loading_seconds"] = model_load["model_loading_seconds"]
                timing["prediction_heads_seconds"] = timing.get("head_seconds")
                window_config = "60/30; batch=2; first-window ownership"
            row = dict(model=name, frames=100, precision="bf16", window_config=window_config,
                       ate_rmse_m=metrics["ate_rmse_m"],
                       adjacent_translation_rmse_m=metrics["adjacent_translation_rmse_m"],
                       adjacent_rotation_rmse_deg=metrics["adjacent_rotation_rmse_deg"],
                       boundary_translation_rmse_m=metrics["boundary_translation_rmse_m"],
                       boundary_rotation_rmse_deg=metrics["boundary_rotation_rmse_deg"],
                       model_loading_seconds=timing.get("model_loading_seconds"),
                       backbone_seconds=timing.get("backbone_seconds"),
                       prediction_heads_seconds=timing.get("prediction_heads_seconds"),
                       forward_seconds=timing["forward_seconds"],
                       stitching_seconds=timing["overlap_stitching_seconds"],
                       reconstruction_total_seconds=timing["reconstruction_total_seconds"],
                       protocol_evaluation_seconds=evaluation_timing["gt_evaluation_seconds"],
                       plotting_seconds=evaluation_timing["plotting_seconds"],
                       file_export_seconds=evaluation_timing["file_export_seconds"],
                       peak_allocated_bytes=timing.get("peak_allocated_bytes", manifest["peak_allocated_bytes"] if name != "vggt_long" else None),
                       peak_reserved_bytes=timing.get("peak_reserved_bytes", manifest["peak_reserved_bytes"] if name != "vggt_long" else None),
                       cpu_peak_rss_bytes=timing.get("cpu_peak_rss_bytes", manifest["cpu_peak_rss_bytes"] if name != "vggt_long" else None))
            rows.append(row)
        raw = compare_raw_windows(args.output)
        provenance["raw_window_differences"] = str(args.output / "raw_window_prediction_differences.json")
        write_comparison(rows, args.output, provenance)
        with (args.output / "comparison_report.md").open("a") as stream:
            stream.write("\n## 拼接前窗口预测差异\n\n")
            stream.write("| Window | Pair | c2w max/mean | depth max/mean | world_points max/mean |\n")
            stream.write("|---:|---|---:|---:|---:|\n")
            for item in raw:
                fields = item["fields"]
                fmt = lambda key: f"{fields[key]['max_abs']:.6g} / {fields[key]['mean_abs']:.6g}"
                stream.write(f"| {item['window']} | {item['pair']} | {fmt('c2w')} | {fmt('depth')} | {fmt('world_points')} |\n")
            stream.write("\n逐帧输入与这些原始局部预测未做 GT 对齐。Long 用后窗口归属，ours 用前窗口归属，最终轨迹差异还包含拼接及归属影响。\n")
            stream.write("\n## 指定边界误差\n\n")
            for row in rows:
                metric = json.loads((args.output / row["model"] / "common_eval/evaluation.json").read_text())
                stream.write(f"### {row['model']}\n\n")
                for boundary in metric["boundary_errors"]:
                    stream.write(f"- {boundary['before']}→{boundary['after']}: translation {boundary['translation_error_m']:.6f} m; rotation {boundary['rotation_error_deg']:.6f} deg\n")
                stream.write("\n")
        write_json(args.output / "COMPLETE.json", dict(status="complete", models=[r["model"] for r in rows]))
        print(json.dumps(rows, indent=2), flush=True)
    except Exception:
        write_json(args.output / "FAILED.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=10)
        if sampler_stream is not None:
            sampler_stream.close()
        lock.close()


if __name__ == "__main__":
    main()
