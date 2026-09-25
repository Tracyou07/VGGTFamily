"""Opt-in one-window-at-a-time execution for v8 independent inference.

The original WindowReconstructor, heads and AlignmentStitcher do the math.
This module changes only when a window is decoded and when its CPU output is
written. In particular, it never holds more than one window's GPU features.
"""
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from experiments.ours_v6.correspondence import save_overlap_mask
from experiments.ours_v6.metrics import summarize
from experiments.ours_v6.runtime import write_json, write_point_cloud
from experiments.ours_v6.windows import make_windows
from experiments.ours_v4.artifacts import evaluate, write_cloud
from vggt.v8.joint_alignment import AlignmentStitcher, JointAlignmentConfig
from vggt.v8.model import WindowReconstructor


DENSE_KEYS = ("c2w", "intrinsics", "depth", "confidence",
              "world_points", "world_points_conf")


def run_independent_window(reconstructor, images, frame_ids, lo, hi,
                           window_size, overlap, query_chunk_size,
                           reuse_image_encoding=False,
                           cache_local_kv_dtype=False,
                           correspondence_attention_path="explicit",
                           dense_head_frame_chunk=None):
    """Decode one original window with the unchanged independent model path."""
    if not 0 <= lo < hi <= len(images) or hi - lo > window_size:
        raise ValueError("invalid independent window slice")
    if len(frame_ids) != len(images):
        raise ValueError("frame ID count differs from input tensor")
    result = reconstructor(
        images[lo:hi], frame_ids[lo:hi], mode="independent",
        window_size=window_size, overlap=overlap,
        query_chunk_size=query_chunk_size,
        reuse_image_encoding=reuse_image_encoding,
        cache_local_kv_dtype=cache_local_kv_dtype,
        correspondence_attention_path=correspondence_attention_path,
        dense_head_frame_chunk=dense_head_frame_chunk,
    )
    if result["windows"] != [(0, hi - lo)] or len(result["predictions"]) != 1:
        raise ValueError("single-window reconstruction returned multiple windows")
    prediction = result["predictions"][0]
    if list(prediction["frame_ids"]) != list(frame_ids[lo:hi]):
        raise ValueError("single-window frame IDs changed")
    return prediction, result["timing"], result["memory"], result["retained_head_features_bytes"]


def numpy_prediction(prediction):
    """Use the same CPU array contract as the unstreamed worker."""
    converted = {key: value.numpy() if torch.is_tensor(value) else value
                 for key, value in prediction.items()}
    if any(torch.is_tensor(value) for value in converted.values()):
        raise ValueError("prediction contains a tensor after CPU conversion")
    return converted


class DenseDiskWriter:
    """Store owned dense predictions on disk until the final NPZ is exported."""
    def __init__(self, scratch, total_frames, example):
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.total_frames = total_frames
        self.count = 0
        self.paths = {}
        self.maps = {}
        for key in DENSE_KEYS:
            value = np.asarray(example[key])
            if value.ndim < 1 or len(value) == 0 or not np.isfinite(value).all():
                raise ValueError(f"invalid dense output {key}")
            path = self.scratch / f"{key}.npy"
            self.paths[key] = path
            self.maps[key] = np.lib.format.open_memmap(
                path, mode="w+", dtype=value.dtype,
                shape=(total_frames, *value.shape[1:]))

    def append(self, transformed, fresh):
        count = len(fresh)
        if not count or self.count + count > self.total_frames:
            raise ValueError("invalid owned-frame count")
        for key in DENSE_KEYS:
            value = np.asarray(transformed[key])[fresh]
            if value.shape != self.maps[key][self.count:self.count + count].shape:
                raise ValueError(f"owned dense shape changed: {key}")
            self.maps[key][self.count:self.count + count] = value
        self.count += count

    def arrays(self):
        if self.count != self.total_frames:
            raise ValueError("owned dense output is incomplete")
        for array in self.maps.values():
            array.flush()
        return self.maps

    def cleanup_success(self):
        if self.count != self.total_frames:
            raise ValueError("cannot remove incomplete dense scratch")
        for array in self.maps.values():
            array.flush()
        self.maps.clear()
        gc.collect()
        for path in self.paths.values():
            path.unlink()
        self.scratch.rmdir()


def execute_streaming(args, model, images, frame_ids, windows, saved,
                      save_npz, manifest, task_start, input_seconds,
                      model_seconds, profile):
    """Complete the old export/evaluation contract while streaming windows."""
    if args.mode != "independent" or not args.stream_independent:
        raise ValueError("streaming is available only for opted-in independent mode")
    if not args.whole_task_measurement:
        raise ValueError("streaming requires whole-task peak measurement")
    if args.reuse_image_encoding:
        raise ValueError("cross-window encoding reuse is unsupported in streaming mode")
    if windows != make_windows(len(images), args.window_size, args.overlap):
        raise ValueError("unexpected window schedule")

    wrapper = WindowReconstructor(model)
    stitch = AlignmentStitcher(args.output / "alignment",
                               JointAlignmentConfig(mode=args.alignment_mode))
    writer = None
    cloud, cloud_windows = [], []
    forward_seconds = stitch_seconds = export_seconds = diagnostic_seconds = 0.0
    backbone_seconds = head_seconds = transfer_seconds = 0.0
    max_head_features = 0
    max_window_memory = {}
    post_window_allocated = []
    progress = args.output / "stream_progress.jsonl"
    for window_id, (lo, hi) in enumerate(windows):
        forward_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, timing, memory, head_bytes = run_independent_window(
                wrapper, images, frame_ids, lo, hi, args.window_size, args.overlap,
                args.query_chunk_size, args.reuse_image_encoding,
                args.cache_local_kv_dtype, args.correspondence_attention_path,
                args.dense_head_frame_chunk)
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - forward_start
        backbone_seconds += timing["backbone_seconds"]
        head_seconds += timing["head_seconds"]
        transfer_seconds += timing["output_transfer_seconds"]
        max_head_features = max(max_head_features, head_bytes)
        for key, value in memory.items():
            if isinstance(value, (int, float)):
                max_window_memory[key] = max(max_window_memory.get(key, 0), value)
        live_bytes = torch.cuda.memory_allocated()
        post_window_allocated.append(live_bytes)

        prediction = numpy_prediction(prediction)
        folder = args.output / "windows" / f"{window_id:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        export_start = time.perf_counter()
        save_npz(folder / "local.npz", **prediction)
        export_seconds += time.perf_counter() - export_start

        if window_id:
            start = time.perf_counter()
            save_overlap_mask(stitch.previous, prediction,
                args.output / "alignment" /
                f"edge_{window_id-1:04d}_{window_id:04d}_correspondence_mask.npz")
            diagnostic_seconds += time.perf_counter() - start
        start = time.perf_counter()
        fresh, transformed = stitch.add(prediction, window_id)
        stitch_seconds += time.perf_counter() - start
        if writer is None:
            writer = DenseDiskWriter(args.output / "_stream_dense", len(frame_ids), transformed)
        start = time.perf_counter()
        writer.append(transformed, fresh)
        export_seconds += time.perf_counter() - start

        start = time.perf_counter()
        xyz, _ = write_point_cloud(folder / "point_head_global.ply",
                                   transformed, images[lo:hi], fresh, window_id)
        write_cloud(folder / "depth_unprojection_global.ply",
                    transformed, images[lo:hi], fresh, 16, window_id)
        cloud.append(xyz)
        cloud_windows.extend([window_id] * len(xyz))
        diagnostic_seconds += time.perf_counter() - start
        with progress.open("a") as stream:
            stream.write(json.dumps(dict(window_id=window_id, frame_range=[lo, hi],
                local_prediction=str(folder / "local.npz"),
                owned_frames=len(fresh), post_window_allocated_bytes=live_bytes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())) + "\n")
        del transformed, prediction, xyz

    global_result = stitch.finish(frame_ids)
    export_start = time.perf_counter()
    save_npz(args.output / "global_trajectory.npz", **global_result)
    dense = writer.arrays()
    if not np.array_equal(dense["c2w"], global_result["c2w"]):
        raise ValueError("streamed c2w ownership disagrees with trajectory")
    if not np.array_equal(dense["intrinsics"], global_result["intrinsics"]):
        raise ValueError("streamed intrinsic ownership disagrees with trajectory")
    save_npz(args.output / "global_predictions.npz",
             frame_ids=global_result["frame_ids"],
             source_window=global_result["source_window"], **dense)
    del dense
    writer.cleanup_success()
    export_seconds += time.perf_counter() - export_start

    start = time.perf_counter()
    evaluate(global_result, saved["scene_root"], args.output)
    summary = summarize(args.output)
    evaluation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = np.concatenate(cloud)
    colors = np.asarray(cloud_windows)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*xyz.T, c=colors, s=.2)
    ax.set_title("Point-head cloud; colors = source window; no GT transform")
    fig.savefig(args.output / "point_head_preview.png", dpi=150)
    plt.close(fig)
    diagnostic_seconds += time.perf_counter() - start

    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    del wrapper, model
    manifest["correspondence"] = dict(pair_count=0,
        interpretation="independent single-window execution; no remote K/V")
    manifest["communication_mode"] = "independent"
    manifest["streaming"] = dict(enabled=True,
        windows_processed=len(windows), max_window_head_features_bytes=max_head_features,
        post_window_allocated_bytes=post_window_allocated,
        gpu_window_states_retained=1,
        cpu_local_predictions_retained="current window for next alignment edge only",
        dense_owned_outputs="temporary disk-backed arrays; removed after successful export",
        no_empty_cache_between_windows=True)
    manifest["optimization"] = dict(
        backend_profile=profile,
        correspondence_query_chunk_size=args.query_chunk_size,
        correspondence_attention_path=args.correspondence_attention_path,
        reuse_image_encoding=args.reuse_image_encoding,
        dense_head_frame_chunk=args.dense_head_frame_chunk,
        cache_local_kv_dtype=args.cache_local_kv_dtype,
        npz_compression_level=args.npz_compression_level,
        whole_task_peak_scope=True, peak_resets=1,
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark)
    prep = saved["preprocessing"]["elapsed_seconds"]
    manifest.update(evaluation_summary=summary,
        timing=dict(backbone_seconds=backbone_seconds, head_seconds=head_seconds,
            output_transfer_seconds=transfer_seconds,
            forward_seconds=forward_seconds,
            overlap_stitching_seconds=stitch_seconds,
            reconstruction_without_historical_preprocessing_seconds=forward_seconds + stitch_seconds,
            reconstruction_total_seconds=prep + forward_seconds + stitch_seconds,
            image_preprocessing_seconds=prep, input_read_seconds=input_seconds,
            model_load_seconds=model_seconds,
            export_seconds=export_seconds, diagnostic_seconds=diagnostic_seconds,
            protocol_evaluation_seconds=evaluation_seconds,
            task_seconds=time.perf_counter() - task_start),
        memory=max_window_memory,
        retained_head_features_bytes=max_head_features,
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        trainable_parameters=0)
    write_json(args.output / "run_manifest.json", manifest)
    write_json(args.output / "COMPLETE.json",
               dict(status="complete", kind="diagnostic",
                    mode="independent", stream_independent=True))
