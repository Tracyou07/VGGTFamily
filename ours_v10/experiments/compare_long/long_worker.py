"""Measurement-only adapter around the existing ScanNet native VGGT-Long backend."""
import argparse
import functools
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

BASE = Path("/home/ubuntu/yjh/feedforwardreconstruct")
SCANNET = BASE / "eval/scannet"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --gpu")
    args.output.mkdir(parents=True, exist_ok=False)
    events = []
    try:
        saved = torch.load(args.input, map_location="cpu", weights_only=True)
        ids = saved["frame_ids"]
        paths = saved["rgb_paths"]
        if len(ids) != 100 or len(paths) != 100:
            raise ValueError("expected exactly the fixed 100 frames")
        sys.path.insert(0, str(SCANNET))
        import scannet_eval.backends.runtime as runtime
        from scannet_eval.backends import create_backend
        runtime.install_source(BASE / "vggtlong", "base_models")
        runtime.install_source(BASE / "vggtlong/base_models", "vggt")
        sys.path.insert(0, str(SCANNET / ".runtime/long_deps"))
        import vggt_long
        from base_models.vggt.models.vggt import VGGT
        from vggt.models.aggregator import Aggregator
        from vggt.heads.camera_head import CameraHead
        from vggt.heads.dpt_head import DPTHead
        from LoopModels.LoopModel import LoopDetector
        import matplotlib.pyplot as plt
        import base_models.vggt.utils.load_fn as loader

        def sync():
            torch.cuda.synchronize()

        def timed(obj, name, category, gpu=False):
            original = getattr(obj, name)
            @functools.wraps(original)
            def call(*a, **kw):
                if gpu: sync()
                start = time.perf_counter()
                try:
                    return original(*a, **kw)
                finally:
                    if gpu: sync()
                    events.append((category, start, time.perf_counter()))
            setattr(obj, name, call)

        for obj, name, cat, gpu in (
            (VGGT, "forward", "forward", True),
            (Aggregator, "forward", "backbone", True),
            (CameraHead, "forward", "camera_head", True),
            (DPTHead, "forward", "dense_heads", True),
            (vggt_long.VGGT_Long, "run", "native_run", True),
            (vggt_long.VGGT_Long, "process_long_sequence", "long_sequence", True),
            (vggt_long.VGGT_Long, "process_single_chunk", "chunk", True),
            (vggt_long.VGGT_Long, "get_loop_pairs", "retrieval", True),
            (LoopDetector, "load_model", "model_load", True),
            (loader, "load_and_preprocess_images", "preprocessing", False),
            (np, "save", "export", False),
            (np, "savetxt", "export", False),
            (vggt_long, "save_confident_pointcloud_batch", "export", False),
            (vggt_long.VGGT_Long, "save_camera_poses", "export", False),
            (plt, "savefig", "plot", False),
        ):
            timed(obj, name, cat, gpu)

        native_instances = []
        native_init = vggt_long.VGGT_Long.__init__
        def record_init(obj, *a, **kw):
            native_init(obj, *a, **kw)
            native_instances.append(obj)
        vggt_long.VGGT_Long.__init__ = record_init

        # Native backend normally deletes these new-run intermediates. Preserve
        # only paths under this exclusive output for raw-window auditing.
        old_remove = runtime.shutil.rmtree
        def preserve(path, *a, **kw):
            if Path(path).resolve().is_relative_to(args.output.resolve()):
                return
            return old_remove(path, *a, **kw)
        runtime.shutil.rmtree = preserve

        source_file = str(Path(vggt_long.__file__).resolve())
        source_lines = Path(source_file).read_text().splitlines()
        def line(prefix):
            return next(i for i, row in enumerate(source_lines, 1) if row.strip().startswith(prefix))
        load_begin = line("self.model.load()")
        load_end = line("self.process_long_sequence()")
        marks = {}
        def trace(frame, event, arg):
            if frame.f_code.co_filename == source_file and frame.f_code.co_name == "run" and event == "line":
                if frame.f_lineno == load_begin:
                    sync(); marks["vg gt load start"] = time.perf_counter()
                elif frame.f_lineno == load_end:
                    sync(); marks["vg gt load end"] = time.perf_counter()
                    sys.settrace(None)
                    return None
            return trace

        config = json.loads((SCANNET / "configs/h20.json").read_text())["models"]["long"]
        if config["chunk_size"] != 60 or config["overlap"] != 30 or Path(config["checkpoint"]).resolve() != Path("/data/yjh/share/pretrained/VGGT-1B/model.safetensors"):
            raise ValueError("native verified Long config diverged")
        torch.cuda.set_device(0)
        sync()
        torch.cuda.reset_peak_memory_stats()
        backend = create_backend("long", config, device="cuda:0")
        scene = SimpleNamespace(frame_ids=[int(x) for x in ids], image_paths=paths)
        sys.settrace(trace)
        try:
            pred = backend.predict(scene, args.output)
        finally:
            sys.settrace(None)
        sync()
        native = native_instances[0]
        expected_ranges = [(0, 60), (30, 90), (60, 100)]
        if [tuple(map(int, row)) for row in native.chunk_indices] != expected_ranges:
            raise ValueError("Long used unexpected chunks")
        chunk_dir = Path(native.result_unaligned_dir)
        checks = []
        local_poses = []
        local_ids = []
        local_owners = []
        shared = saved["images"].numpy()
        for i, (start, end) in enumerate(expected_ranges):
            chunk_path = chunk_dir / f"chunk_{i}.npy"
            chunk = np.load(chunk_path, allow_pickle=True).item() # trusted, freshly emitted native file
            image = np.asarray(chunk["images"])
            local_poses.append(np.asarray(chunk["extrinsic"]))
            local_ids.extend(ids[start:end])
            local_owners.extend([i] * (end - start))
            target = shared[start:end]
            difference = float(np.max(np.abs(image - target)))
            checks.append(dict(chunk=i, start=start, end=end, max_input_abs_diff=difference,
                               file=str(chunk_path), sha256=_sha(chunk_path)))
            if image.shape != target.shape or difference > 1e-6:
                raise ValueError(f"Long preprocessed input differs from shared tensor in chunk {i}: {difference}")
        np.savez_compressed(args.output / "pre_stitch_trajectory.npz",
                            frame_ids=np.asarray(local_ids), c2w=np.concatenate(local_poses),
                            source_window=np.asarray(local_owners))
        meta = dict(pred.metadata)
        meta["temporary_artifacts_removed"] = False
        meta["preserved_by_compare_long"] = True
        meta["chunk_indices"] = expected_ranges
        meta["input_checks"] = checks
        meta["checkpoint_sha256"] = _sha(config["checkpoint"])
        meta["frame_list_sha256"] = saved["frame_list_sha256"]
        meta["native_source_path"] = str(source_file)
        meta["native_source_sha256"] = _sha(source_file)
        source_paths = [
            BASE / "vggtlong/vggt_long.py",
            BASE / "vggtlong/base_models/base_model.py",
            BASE / "vggtlong/base_models/vggt/models/vggt.py",
            BASE / "vggtlong/base_models/vggt/models/aggregator.py",
            BASE / "vggtlong/base_models/vggt/heads/camera_head.py",
            BASE / "vggtlong/base_models/vggt/heads/dpt_head.py",
            BASE / "vggtlong/loop_utils/sim3utils.py",
            BASE / "vggtlong/configs/base_config.yaml",
        ]
        meta["source_file_sha256"] = {str(path): _sha(path) for path in source_paths}
        (args.output / "long_manifest.json").write_text(json.dumps(meta, indent=2))
        np.savez_compressed(args.output / "global_trajectory.npz",
                            frame_ids=np.asarray(ids), c2w=pred.poses_c2w,
                            source_window=np.asarray(meta["frame_owner"]))
        cloud = pred.points[::max(1, len(pred.points) // 150000)]
        with (args.output / "point_cloud_preview.ply").open("x") as stream:
            stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(cloud)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
            np.savetxt(stream, cloud, fmt="%.7g")

        def duration(category):
            return sum(end - start for kind, start, end in events if kind == category)
        run_spans = [(start, end) for kind, start, end in events if kind == "native_run"]
        if len(run_spans) != 1:
            raise RuntimeError("expected exactly one native Long run")
        run_start, run_end = run_spans[0]
        def inside(category, lo=run_start, hi=run_end):
            return sum(max(0.0, min(end, hi) - max(start, lo))
                       for kind, start, end in events if kind == category)
        native_vggt_load = marks["vg gt load end"] - marks["vg gt load start"]
        model_load = duration("model_load") + native_vggt_load
        # Native load and exports are excluded only to the extent that they
        # actually overlap native.run(); retrieval model load in __init__ is
        # outside the measured run and must not be subtracted again.
        run_seconds = run_end - run_start
        reconstruction = run_seconds - native_vggt_load - inside("model_load") - inside("export") - inside("plot")
        forward = duration("forward")
        backbone = duration("backbone")
        heads = duration("camera_head") + duration("dense_heads")
        chunk = duration("chunk")
        sequence_spans = [(start, end) for kind, start, end in events if kind == "long_sequence"]
        stitching = sum(end - start - inside("chunk", start, end)
                        - inside("export", start, end) - inside("plot", start, end)
                        for start, end in sequence_spans)
        stitching = max(0.0, stitching)
        timing = dict(model_loading_seconds=model_load,
                      backbone_seconds=backbone, prediction_heads_seconds=heads,
                      forward_seconds=forward, overlap_stitching_seconds=stitching,
                      reconstruction_total_seconds=reconstruction,
                      input_preprocessing_seconds=duration("preprocessing"),
                      retrieval_seconds=duration("retrieval"),
                      unclassified_reconstruction_seconds=max(0.0, reconstruction - forward - stitching),
                      timing_notes="CUDA synchronized around native model forward, aggregator, heads, run and chunk. Native run includes retrieval, repeated chunk preprocessing, transfers and alignment. Model load/export subtracted; GT evaluation and new preview outside run.",
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        (args.output / "timing.json").write_text(json.dumps(timing, indent=2))
        (args.output / "COMPLETE.json").write_text(json.dumps(dict(status="complete")))
    except Exception:
        (args.output / "FAILED.json").write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=2))
        raise


if __name__ == "__main__":
    main()
