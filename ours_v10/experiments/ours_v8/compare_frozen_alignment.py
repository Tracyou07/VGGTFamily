"""CPU-only comparison of v8 alignment modes on frozen window predictions."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch

from experiments.ours_v4.artifacts import evaluate
from experiments.ours_v6.metrics import summarize
from experiments.ours_v6.runtime import sha256, source_identity, write_json
from vggt.v8.joint_alignment import (AlignmentStitcher, JointAlignmentConfig,
                                     loss_components)


def load_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def edge_diagnostics(directory, predictions, mode):
    rows = []
    for edge_index, (left, right) in enumerate(zip(predictions, predictions[1:])):
        path = directory / f"edge_{edge_index:04d}_{edge_index + 1:04d}.json"
        record = json.loads(path.read_text())
        if record.get("status") != "success":
            raise RuntimeError(f"failed edge record: {path}")
        transform_record = record["adjacent"]
        from experiments.ours_v3.geometry import Sim3
        transform = Sim3(transform_record["scale"],
                         np.asarray(transform_record["rotation"], dtype=np.float64),
                         np.asarray(transform_record["translation"], dtype=np.float64))
        diagnostic_config = JointAlignmentConfig(mode="point_camera_joint")
        final_all_components = loss_components(transform, left, right, diagnostic_config)
        if mode == "point_legacy":
            initial = final_all_components
            final = final_all_components
            optimization = {"kind": "legacy result is the initializer; no secondary optimization"}
        else:
            initial = record["initial_loss"]
            final = record["final_loss"]
            optimization = record["optimization"]
        rows.append(dict(
            edge=f"{edge_index:04d}->{edge_index + 1:04d}",
            common_frame_ids=record["common_frame_ids"],
            pairs=record["pairs"],
            adjacent=transform_record,
            initial_loss=initial,
            final_loss=final,
            final_all_component_diagnostics=final_all_components,
            optimization=optimization,
        ))
    return rows


def execute(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    manifest = json.loads(args.source_manifest.read_text())
    window_paths = sorted(args.predictions.glob("*/local.npz"))
    expected_windows = [tuple(values) for values in manifest["windows"]]
    if len(window_paths) != len(expected_windows):
        raise ValueError("frozen window count disagrees with manifest")
    predictions = [load_prediction(path) for path in window_paths]
    for prediction, (lo, hi) in zip(predictions, expected_windows):
        if list(prediction["frame_ids"]) != manifest["frame_ids"][lo:hi]:
            raise ValueError("frozen frame IDs disagree with manifest")
    saved = torch.load(Path(manifest["configuration"]["input"]), map_location="cpu",
                       weights_only=True)
    if list(saved["frame_ids"][:len(manifest["frame_ids"])]) != manifest["frame_ids"]:
        raise ValueError("prepared input frame IDs disagree with manifest")

    config = JointAlignmentConfig(mode=args.mode)
    stitcher = AlignmentStitcher(args.output / "alignment", config)
    edge_times = []
    for window_id, prediction in enumerate(predictions):
        edge_start = time.perf_counter()
        fresh, transformed = stitcher.add(prediction, window_id)
        edge_times.append(time.perf_counter() - edge_start)
        expected_fresh = list(range(len(prediction["frame_ids"]))) if window_id == 0 else \
            list(range(expected_windows[window_id - 1][1] - expected_windows[window_id][0],
                       len(prediction["frame_ids"])))
        if fresh != expected_fresh:
            raise ValueError("front-window ownership changed")
        del transformed
    global_result = stitcher.finish(manifest["frame_ids"])
    np.savez_compressed(args.output / "global_trajectory.npz", **global_result)
    alignment_seconds = float(sum(edge_times))
    diagnostic_start = time.perf_counter()
    edges = edge_diagnostics(args.output / "alignment", predictions, args.mode)
    diagnostic_seconds = time.perf_counter() - diagnostic_start
    evaluation_start = time.perf_counter()
    evaluate(global_result, saved["scene_root"], args.output)
    metrics = summarize(args.output)
    evaluation_seconds = time.perf_counter() - evaluation_start
    result = dict(
        status="success",
        prediction_set=args.prediction_set,
        alignment_mode=args.mode,
        configuration=config.__dict__,
        ownership="front-window first",
        transform_direction="B_local -> A_local",
        alignment_seconds=alignment_seconds,
        per_window_add_seconds=edge_times,
        diagnostic_seconds=diagnostic_seconds,
        evaluation_seconds=evaluation_seconds,
        process_wall_seconds=time.perf_counter() - start,
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        trajectory_metrics=metrics,
        edges=edges,
    )
    write_json(args.output / "alignment_comparison.json", result)
    write_json(args.output / "source_manifest.json", dict(
        source=source_identity(),
        source_run_manifest=str(args.source_manifest),
        source_run_manifest_sha256=sha256(args.source_manifest),
        frozen_prediction_files=[dict(path=str(path), sha256=sha256(path)) for path in window_paths],
        prepared_input=str(manifest["configuration"]["input"]),
        prepared_input_sha256=manifest["input_sha256"],
        source_checkpoint=manifest["checkpoint"],
        source_checkpoint_sha256=manifest["checkpoint_sha256"],
        no_model_forward=True,
        gt_used_only_after_stitching=True,
    ))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--prediction-set", choices=("independent", "overlap_correspondence"), required=True)
    parser.add_argument("--mode", choices=("point_legacy", "point_normalized_control", "point_camera_joint"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        execute(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / "FAILED.json", dict(reason=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
