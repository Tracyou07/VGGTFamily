"""CPU-only v10 stitching experiment using unchanged v9 alignment mathematics."""
import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import json
from pathlib import Path
import resource
import os
import threading
import time
import traceback

import numpy as np
import torch

from experiments.ours_v3.geometry import Sim3
from experiments.ours_v4.artifacts import evaluate
from experiments.ours_v6.metrics import summarize
from experiments.ours_v6.windows import make_windows
from experiments.ours_v6.runtime import sha256, source_identity, write_json
from vggt.v8.joint_alignment import JointAlignmentConfig, loss_components
import vggt.v8.joint_alignment as v8_joint
from vggt.v9.sparse_alignment import (
    SparseAlignmentConfig, V9AlignmentStitcher,
    select_sparse_correspondences,
)


@contextmanager
def timed_full_alignment():
    """Time calls inside unchanged v8 alignment; restore every function."""
    names = {"prepare": "_prepare", "initialization": "legacy_align_overlap",
             "optimization": "_optimize", "full_alignment": "align_overlap_joint"}
    originals = {label: getattr(v8_joint, name) for label, name in names.items()}
    durations = {label: 0. for label in names}

    def wrapper(label):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return originals[label](*args, **kwargs)
            finally:
                durations[label] += time.perf_counter() - start
        return call

    try:
        for label, name in names.items():
            setattr(v8_joint, name, wrapper(label))
        yield durations
    finally:
        for label, name in names.items():
            setattr(v8_joint, name, originals[label])


def load_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def edge_transform(record):
    data = record["adjacent"]
    return Sim3(data["scale"], np.asarray(data["rotation"], dtype=np.float64),
                np.asarray(data["translation"], dtype=np.float64))


@contextmanager
def sampled_incremental_rss(interval_seconds=.005):
    """Peak RSS above edge-start baseline, sampled only during alignment.add."""
    def rss():
        with open("/proc/self/statm") as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    baseline = rss()
    stats = dict(baseline_bytes=baseline, peak_bytes=baseline,
                 incremental_peak_bytes=0, sampling_interval_seconds=interval_seconds,
                 samples=1)
    done = threading.Event()
    def sample():
        while not done.wait(interval_seconds):
            stats["peak_bytes"] = max(stats["peak_bytes"], rss())
            stats["samples"] += 1
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield stats
    finally:
        done.set(); thread.join()
        stats["peak_bytes"] = max(stats["peak_bytes"], rss())
        stats["incremental_peak_bytes"] = max(0, stats["peak_bytes"]-baseline)


def heldout_pixels(a, b, selected_indices, transform, grid_size=16):
    """One distinct, valid pixel per occupied cell; never used for fitting."""
    ids_a, ids_b = list(a["frame_ids"]), list(b["frame_ids"])
    set_b = set(ids_b)
    common = [frame for frame in ids_a if frame in set_b]
    selected = set((str(frame), int(row), int(col))
                   for frame, row, col in selected_indices)
    source, target = [], []
    threshold = .1 * min(
        np.median(a["world_points_conf"][[ids_a.index(frame) for frame in common]]),
        np.median(b["world_points_conf"][[ids_b.index(frame) for frame in common]]))
    for frame in common:
        index_a, index_b = ids_a.index(frame), ids_b.index(frame)
        ap, bp = a["world_points"][index_a], b["world_points"][index_b]
        ac = a["world_points_conf"][index_a]
        bc = b["world_points_conf"][index_b]
        valid = ((ac > threshold) & (bc > threshold) &
                 np.isfinite(ap).all(axis=-1) & np.isfinite(bp).all(axis=-1))
        height, width = valid.shape
        for grid_row in range(grid_size):
            row_start = grid_row * height // grid_size
            row_end = (grid_row + 1) * height // grid_size
            for grid_col in range(grid_size):
                col_start = grid_col * width // grid_size
                col_end = (grid_col + 1) * width // grid_size
                indices = np.argwhere(valid[row_start:row_end, col_start:col_end])
                for dr, dc in indices:
                    row, col = row_start + int(dr), col_start + int(dc)
                    if (str(frame), row, col) not in selected:
                        source.append(bp[row, col]); target.append(ap[row, col])
                        break
    if not source:
        return dict(count=0, mean=None, rmse=None, max=None)
    residual = np.linalg.norm(transform.apply(np.asarray(source, dtype=np.float64)) -
                              np.asarray(target, dtype=np.float64), axis=1)
    return dict(count=len(residual), mean=float(residual.mean()),
                rmse=float(np.sqrt(np.mean(residual**2))), max=float(residual.max()))


def execute(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    source = json.loads(args.source_manifest.read_text())
    paths = sorted(args.predictions.glob("*/local.npz"))
    windows = [tuple(pair) for pair in source["windows"]]
    expected_ids = list(source["frame_ids"])
    configuration = source["configuration"]
    expected_windows = make_windows(len(expected_ids),
        int(configuration["window_size"]), int(configuration["overlap"]))
    if windows != expected_windows or len(paths) != len(windows):
        raise ValueError("frozen prediction window set is incomplete or inconsistent")
    if source.get("dataset") == "Virtual KITTI 1.3.1":
        from experiments.ours_v9.vkitti_131 import scene20_windows
        if windows != scene20_windows(expected_ids,
            int(configuration["window_size"]), int(configuration["overlap"])):
            raise ValueError("Virtual KITTI frame/window order mismatch")
    if args.prediction_set not in ("independent", "camera_only", "overlap_correspondence", "camera_global_overlap"):
        raise ValueError("unknown frozen prediction set")
    if args.predictions.parent.name != args.prediction_set:
        raise ValueError("prediction path and label disagree")

    config = (JointAlignmentConfig(mode="point_camera_joint") if
              args.mode == "point_camera_joint" else SparseAlignmentConfig())
    stitcher = V9AlignmentStitcher(args.output / "alignment", config)
    rows = []
    previous = None
    loading_seconds = alignment_seconds = diagnostic_seconds = 0.
    for window_id, (path, (lo, hi)) in enumerate(zip(paths, windows)):
        phase = time.perf_counter()
        prediction = load_prediction(path)
        prediction_load_seconds = time.perf_counter() - phase
        loading_seconds += prediction_load_seconds
        if list(prediction["frame_ids"]) != expected_ids[lo:hi]:
            raise ValueError(f"frame IDs disagree with frozen manifest: {path}")
        edge_start = time.perf_counter()
        with sampled_incremental_rss() as edge_rss:
            if args.mode == "point_camera_joint":
                with timed_full_alignment() as stage:
                    fresh, transformed = stitcher.add(prediction, window_id)
            else:
                stage = {}
                fresh, transformed = stitcher.add(prediction, window_id)
        add_seconds = time.perf_counter() - edge_start
        alignment_seconds += add_seconds
        expected_fresh = list(range(hi - lo)) if window_id == 0 else list(
            range(windows[window_id - 1][1] - lo, hi - lo))
        if fresh != expected_fresh:
            raise ValueError("front-window ownership changed")
        del transformed
        if window_id:
            edge_path = (args.output / "alignment" /
                         f"edge_{window_id-1:04d}_{window_id:04d}.json")
            record = json.loads(edge_path.read_text())
            if record.get("status") != "success":
                raise RuntimeError(f"failed alignment edge: {edge_path}")
            transform = edge_transform(record)
            phase = time.perf_counter()
            if args.mode == "point_camera_joint":
                selected = select_sparse_correspondences(previous, prediction,
                                                         SparseAlignmentConfig())
                selected_indices = selected.indices
                selected_count = len(selected.source)
                selection_hash = selected.index_sha256
                candidates_per_frame = selected.candidate_per_frame
                selected_per_frame = selected.selected_per_frame
                quadrants_per_frame = selected.quadrants_per_frame
            else:
                selected_indices = [(entry["frame_id"], entry["row"], entry["col"])
                                    for entry in record["selected_pixels"]]
                selected_count = record["selected_pairs"]
                selection_hash = record["index_sha256"]
                candidates_per_frame = record["candidate_per_frame"]
                selected_per_frame = record.get("selected_per_frame", [])
                quadrants_per_frame = record.get("quadrants_per_frame", [])
            all_points = loss_components(transform, previous, prediction,
                                         JointAlignmentConfig(mode="point_camera_joint"))
            heldout = heldout_pixels(previous, prediction, selected_indices, transform)
            diagnostic_seconds += time.perf_counter() - phase
            sparse_stage = record.get("timing_seconds", {})
            rows.append(dict(edge_index=window_id-1,
                left_window=window_id-1, right_window=window_id,
                common_frame_ids=record["common_frame_ids"],
                candidate_per_frame=candidates_per_frame,
                selected_per_frame=selected_per_frame,
                quadrants_per_frame=quadrants_per_frame,
                selected_pairs=selected_count,
                fit_pairs=record["pairs"],
                index_sha256=selection_hash,
                fallback=bool(record.get("fallback", False)),
                fallback_reason=record.get("fallback_reason"),
                selection_seconds=sparse_stage.get("selection", 0.),
                prepare_seconds=stage.get("prepare", 0.),
                initialization_seconds=(sparse_stage.get("initialization", 0.) if
                                        args.mode != "point_camera_joint" else
                                        stage["initialization"]),
                optimization_seconds=(sparse_stage.get("optimization", 0.) if
                                      args.mode != "point_camera_joint" else
                                      stage["optimization"]),
                full_fallback_seconds=sparse_stage.get("fallback_full_seconds", 0.),
                fit_seconds=(sparse_stage.get("total_alignment", 0.) if
                             args.mode != "point_camera_joint" else stage["full_alignment"]),
                prediction_load_seconds=prediction_load_seconds,
                add_seconds=add_seconds,
                alignment_incremental_peak_rss_bytes=edge_rss["incremental_peak_bytes"],
                alignment_rss_baseline_bytes=edge_rss["baseline_bytes"],
                alignment_rss_sampled_peak_bytes=edge_rss["peak_bytes"],
                alignment_rss_sampling_interval_seconds=edge_rss["sampling_interval_seconds"],
                alignment_rss_samples=edge_rss["samples"],
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                scale=transform.scale,
                adjacent=record["adjacent"],
                global_transform=record["global_transform"],
                fit_final_loss=record["final_loss"],
                all_point_diagnostics=all_points,
                other_pixel_diagnostics=heldout,
                other_pixels_independent_of_fit=(args.mode != "point_camera_joint" and
                                                 not record.get("fallback", False))))
            print(f"{args.prediction_set}/{args.mode} edge {window_id:02d}/{len(windows)-1:02d} "
                  f"fit={rows[-1]['fit_pairs']} add={add_seconds:.3f}s "
                  f"fallback={rows[-1]['fallback']}", flush=True)
            with (args.output / "edge_progress.jsonl").open("a") as stream:
                stream.write(json.dumps(rows[-1]) + "\n")
        previous = prediction

    global_result = stitcher.finish(expected_ids)
    phase = time.perf_counter()
    np.savez_compressed(args.output / "global_trajectory.npz", **global_result)
    export_seconds = time.perf_counter() - phase
    phase = time.perf_counter()
    if source.get("dataset") == "Virtual KITTI 1.3.1":
        from experiments.ours_v9.vkitti_131 import evaluate_trajectory
        if args.vkitti_raw_root is None or args.vkitti_condition != source["condition"]:
            raise ValueError("Virtual KITTI raw root/condition required")
        metrics = evaluate_trajectory(global_result, args.vkitti_raw_root,
            args.vkitti_condition, args.output, expected_frames=len(expected_ids),
            require_full=len(expected_ids)==837)
    else:
        saved = torch.load(source["configuration"]["input"], map_location="cpu",
                           weights_only=True)
        if list(saved["frame_ids"][:len(expected_ids)]) != expected_ids:
            raise ValueError("prepared input frame IDs differ from frozen run")
        evaluate(global_result, saved["scene_root"], args.output)
        metrics = summarize(args.output)
    evaluation_seconds = time.perf_counter() - phase
    if len(rows) != len(windows)-1 or metrics["ownership_boundaries"]["count"] != len(windows)-1:
        raise ValueError("alignment/ownership boundary count mismatch")
    boundaries = json.loads((args.output / "boundary_diagnostics.json").read_text())
    for row, boundary in zip(rows, boundaries):
        row["boundary_translation_error_m"] = boundary["translation_error"]
        row["boundary_rotation_error_deg"] = boundary["rotation_error_deg"]
    result = dict(status="success", prediction_set=args.prediction_set,
        alignment_mode=args.mode, source_commit=source["sources"]["commit"],
        configuration=asdict(config) if args.mode != "point_camera_joint" else config.__dict__,
        loading_seconds=loading_seconds, alignment_seconds=alignment_seconds,
        diagnostic_seconds=diagnostic_seconds, export_seconds=export_seconds,
        evaluation_seconds=evaluation_seconds,
        process_wall_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        fallback_edges=[row["edge_index"] for row in rows if row["fallback"]],
        trajectory_metrics=metrics, edges=rows)
    write_json(args.output / "alignment_comparison.json", result)
    with (args.output / "edge_details.csv").open("w", newline="") as stream:
        fields = ["edge_index", "common_frame_ids", "candidate_per_frame",
                  "selected_pairs", "fit_pairs", "index_sha256", "fallback",
                  "fallback_reason", "selection_seconds", "prepare_seconds",
                  "initialization_seconds", "optimization_seconds",
                  "full_fallback_seconds", "fit_seconds", "add_seconds",
                  "prediction_load_seconds", "point_residual_rmse",
                  "camera_center_residual_mean", "camera_rotation_residual_mean_rad",
                  "alignment_incremental_peak_rss_bytes", "process_peak_rss_bytes",
                  "boundary_translation_error_m", "boundary_rotation_error_deg",
                  "selected_per_frame", "quadrants_per_frame",
                  "scale", "other_pixel_count",
                  "other_pixel_rmse", "other_pixels_independent_of_fit"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(edge_index=row["edge_index"],
                common_frame_ids=json.dumps(row["common_frame_ids"]),
                candidate_per_frame=json.dumps(row["candidate_per_frame"]),
                selected_pairs=row["selected_pairs"], fit_pairs=row["fit_pairs"],
                index_sha256=row["index_sha256"], fallback=row["fallback"],
                fallback_reason=row["fallback_reason"],
                selection_seconds=row["selection_seconds"],
                prepare_seconds=row["prepare_seconds"],
                initialization_seconds=row["initialization_seconds"],
                optimization_seconds=row["optimization_seconds"],
                full_fallback_seconds=row["full_fallback_seconds"],
                fit_seconds=row["fit_seconds"], add_seconds=row["add_seconds"],
                prediction_load_seconds=row["prediction_load_seconds"],
                point_residual_rmse=row["all_point_diagnostics"]["point_residual_rmse"],
                camera_center_residual_mean=row["all_point_diagnostics"]["center_residual_mean"],
                camera_rotation_residual_mean_rad=row["all_point_diagnostics"]["rotation_residual_mean_rad"],
                alignment_incremental_peak_rss_bytes=row["alignment_incremental_peak_rss_bytes"],
                process_peak_rss_bytes=row["process_peak_rss_bytes"],
                boundary_translation_error_m=row["boundary_translation_error_m"],
                boundary_rotation_error_deg=row["boundary_rotation_error_deg"],
                selected_per_frame=json.dumps(row["selected_per_frame"]),
                quadrants_per_frame=json.dumps(row["quadrants_per_frame"]),
                scale=row["scale"],
                other_pixel_count=row["other_pixel_diagnostics"]["count"],
                other_pixel_rmse=row["other_pixel_diagnostics"]["rmse"],
                other_pixels_independent_of_fit=row["other_pixels_independent_of_fit"]))
    write_json(args.output / "source_manifest.json", dict(
        current_source=source_identity(), source_run_manifest=str(args.source_manifest),
        source_run_manifest_sha256=sha256(args.source_manifest),
        frozen_prediction_files=[dict(path=str(path), sha256=sha256(path))
                                 for path in paths],
        input_path=source["configuration"]["input"],
        input_sha256=source["input_sha256"],
        checkpoint=source["checkpoint"],
        checkpoint_sha256=source["checkpoint_sha256"],
        no_model_forward=True, gt_used_only_after_stitching=True))
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(args.output / "alignment_comparison.json", result)
    write_json(args.output / "COMPLETE.json", dict(status="complete", mode=args.mode,
                                                   prediction_set=args.prediction_set))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prediction-set", choices=("independent", "camera_only", "overlap_correspondence", "camera_global_overlap"),
                        required=True)
    parser.add_argument("--mode", choices=("point_camera_joint",
                                           "sparse_point_camera_joint"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vkitti-raw-root", type=Path)
    parser.add_argument("--vkitti-condition", choices=("clone", "rain", "fog"))
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        execute(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "FAILED.json", dict(
            reason=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
