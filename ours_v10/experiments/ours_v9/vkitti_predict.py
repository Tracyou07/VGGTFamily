"""Freeze v8 local window predictions for Virtual KITTI 1.3.1 Scene20.

The public invocation preprocesses once, then launches two fresh CUDA workers
that load the same immutable CPU tensor. It never aligns, evaluates or exports
point clouds. --smoke-frames is an explicitly shortened interface test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from experiments.ours_v7.backend_profiles import early_prepare
_EARLY_PROFILE = early_prepare(sys.argv[1:]) if "--worker-mode" in sys.argv else None

import numpy as np
import torch

from experiments.ours_v6.runtime import ROOT, preflight, sha256, source_identity, write_json
from experiments.ours_v9.vkitti_131 import DEFAULT_RAW, EVAL_SOURCE, inspect_condition, scene20_windows

MODES = ("independent", "overlap_correspondence")
KEYS = ("frame_ids", "c2w", "intrinsics", "depth", "world_points", "world_points_conf")


def tensor_sha256(images):
    array = images.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def prediction_values(prediction, frame_ids):
    values = {key: (prediction[key].numpy() if torch.is_tensor(prediction[key])
                    else prediction[key]) for key in KEYS}
    if list(values["frame_ids"]) != list(frame_ids):
        raise ValueError("prediction frame IDs changed")
    if not all(np.isfinite(value).all() for key, value in values.items()
               if key != "frame_ids"):
        raise ValueError("nonfinite local prediction")
    return values


def _identity(inventory):
    return dict(rgb_sources=list(inventory.sources),
        gt_source_used_for_validation_only=True,
        evaluator_source=str(EVAL_SOURCE),
        evaluator_sha256={name:sha256(EVAL_SOURCE/"virtual_kitti_eval"/name)
            for name in ("data.py", "metrics.py", "config.py")})


def worker_paths(root, mode):
    if mode not in MODES:
        raise ValueError("invalid worker mode")
    root = Path(root)
    return root / "input_manifest.json", root / mode


def prepare(args):
    inventory = inspect_condition(args.raw_root, args.condition,
        expected_frames=args.frames, require_full=args.frames==837)
    if args.output.exists():
        raise FileExistsError(args.output)
    # A failed preparation still has a directory for FAILED.json.
    args.output.mkdir(parents=True)
    from vggt.utils.load_fn import load_and_preprocess_images
    started = time.perf_counter()
    images = load_and_preprocess_images([str(path) for path in inventory.image_paths])
    if images.ndim != 4 or len(images) != args.frames or not torch.isfinite(images).all():
        raise ValueError("invalid preprocessed image tensor")
    image_hash = tensor_sha256(images)
    saved = dict(images=images, frame_ids=list(inventory.frame_ids),
                 rgb_paths=[str(path) for path in inventory.image_paths],
                 preprocessing=dict(loader="original VGGT load_and_preprocess_images crop",
                                    shape=list(images.shape), dtype=str(images.dtype),
                                    elapsed_seconds=time.perf_counter()-started))
    input_path = args.output / "inputs.pt"
    with input_path.open("xb") as stream:
        torch.save(saved, stream)
    input_hash = sha256(input_path)
    checkpoint = Path(json.loads((ROOT/"configs/v7_validation.json").read_text())["checkpoint"])
    manifest = dict(dataset="Virtual KITTI 1.3.1", scene="Scene20",
        condition=args.condition, frame_ids=list(inventory.frame_ids),
        windows=scene20_windows(inventory.frame_ids), frame_count=args.frames,
        smoke=args.frames!=837, preprocessing=saved["preprocessing"],
        image_tensor_sha256=image_hash, input_sha256=input_hash,
        input_path=str(input_path.resolve()), checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), source_identity=source_identity(),
        data_identity=_identity(inventory),
        prediction_input_fields=["images", "frame_ids"],
        gt_not_serialized_in_input=True)
    write_json(args.output / "input_manifest.json", manifest)
    return manifest


def worker(args):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.v8.model import WindowReconstructor
    from experiments.ours_v7.backend_profiles import apply_backend_profile, snapshot
    from experiments.ours_v7.diagnostic_export import save_npz_fast
    from experiments.ours_v8.stream_independent import run_independent_window
    if _EARLY_PROFILE != "native_vggt":
        raise ValueError("native backend must be configured before CUDA initialization")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must match physical --gpu")
    preflight_result = preflight(args.gpu, args.output.parent, min_disk_gib=5)
    apply_backend_profile("native_vggt", torch)
    torch.cuda.init(); torch.cuda.reset_peak_memory_stats()  # once, before model load
    torch.manual_seed(2026); np.random.seed(2026)
    input_manifest, mode_output = worker_paths(args.output, args.worker_mode)
    source = json.loads(input_manifest.read_text())
    if source["frame_count"] != args.frames or source["condition"] != args.condition:
        raise ValueError("prepared input metadata mismatch")
    input_path = Path(source["input_path"])
    if sha256(input_path) != source["input_sha256"]:
        raise ValueError("prepared input file changed")
    saved = torch.load(input_path, map_location="cpu", weights_only=True)
    images, ids = saved["images"], saved["frame_ids"]
    if ids != source["frame_ids"] or tensor_sha256(images) != source["image_tensor_sha256"]:
        raise ValueError("prepared image tensor/frame order changed")
    checkpoint = Path(source["checkpoint"])
    if sha256(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    identity = source_identity()
    if identity["status"]:
        raise RuntimeError("prediction requires a committed, clean worktree")
    mode_output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model = VGGT().eval().requires_grad_(False)
    weights = load_file(str(checkpoint)); model.load_state_dict(weights, strict=True)
    del weights
    model.cuda()
    windows = scene20_windows(ids)
    records = []
    reconstructor = WindowReconstructor(model)
    forward_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if args.worker_mode == "independent":
            predictions = []
            for lo, hi in windows:
                prediction, _, _, _ = run_independent_window(
                    reconstructor, images, ids, lo, hi, 60, 10, 512,
                    reuse_image_encoding=False, cache_local_kv_dtype=True,
                    correspondence_attention_path="native_sdpa",
                    dense_head_frame_chunk=None)
                predictions.append(prediction)
        else:
            result = reconstructor(images, ids,
                mode="overlap_correspondence", window_size=60, overlap=10,
                query_chunk_size=512, reuse_image_encoding=False,
                cache_local_kv_dtype=True, correspondence_attention_path="native_sdpa",
                dense_head_frame_chunk=None)
            if result["windows"] != windows:
                raise ValueError("v8 returned wrong windows")
            predictions = result["predictions"]
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_start
    for index, (prediction, (lo, hi)) in enumerate(zip(predictions, windows)):
        values = prediction_values(prediction, ids[lo:hi])
        folder = mode_output / "windows" / f"{index:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "local.npz"
        save_npz_fast(path, **values)
        records.append(dict(window=index, lo=lo, hi=hi, path=str(path),
                            sha256=sha256(path), bytes=path.stat().st_size,
                            fields=list(values)))
        if args.worker_mode == "independent":
            predictions[index] = None
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    manifest = dict(dataset="Virtual KITTI 1.3.1", scene="Scene20",
        condition=args.condition, communication_mode=args.worker_mode,
        frame_ids=ids, windows=windows, sources=identity,
        preprocessing=saved["preprocessing"], input_sha256=source["input_sha256"],
        image_tensor_sha256=source["image_tensor_sha256"],
        checkpoint=str(checkpoint), checkpoint_sha256=source["checkpoint_sha256"],
        precision="bf16", configuration=dict(input=str(input_path), frames=args.frames,
            window_size=60, overlap=10, backend_profile="native_vggt",
            correspondence_attention_path="native_sdpa", query_chunk_size=512,
            cache_local_kv_dtype=True, dense_head_frame_chunk=None,
            reuse_image_encoding=False),
        backend_profile=dict(requested="native_vggt", effective=snapshot(torch)),
        preflight=preflight_result, gpu_uuid=preflight_result["gpu_uuid"],
        prediction_files=records,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        peak_scope="one fresh process from CUDA initialization through local.npz export; never reset during run",
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        forward_seconds=forward_seconds, process_seconds=time.perf_counter()-start,
        no_alignment_or_gt_evaluation=True, no_point_cloud_export=True)
    write_json(mode_output/"run_manifest.json", manifest)
    write_json(mode_output/"COMPLETE.json", dict(status="complete", mode=args.worker_mode))


def execute(args):
    # Preflight before creating files; no other task is interrupted.
    preflight(args.gpu, args.output.parent, min_disk_gib=5)
    prepare(args)
    for mode in MODES:
        command = [sys.executable, "-B", "-u", "-m", "experiments.ours_v9.vkitti_predict",
                   "--condition", args.condition, "--gpu", str(args.gpu),
                   "--output", str(args.output), "--raw-root", str(args.raw_root),
                   "--smoke-frames", str(args.frames), "--worker-mode", mode,
                   "--backend-profile", "native_vggt"]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        with (args.output/f"{mode}.log").open("x") as log:
            status = subprocess.run(command, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            raise RuntimeError(f"prediction worker failed: {mode}; see its log")
    from experiments.ours_v9.vkitti_align import validate_prediction_pair
    left, right = [json.loads((args.output/m/"run_manifest.json").read_text())
                   for m in MODES]
    validate_prediction_pair(left, right)
    write_json(args.output/"run_manifest.json", dict(
        dataset="Virtual KITTI 1.3.1", scene="Scene20", condition=args.condition,
        frame_count=args.frames, windows=left["windows"], smoke=args.frames!=837,
        input_sha256=left["input_sha256"], image_tensor_sha256=left["image_tensor_sha256"],
        checkpoint_sha256=left["checkpoint_sha256"],
        mode_manifests=[str(args.output/m/"run_manifest.json") for m in MODES]))
    write_json(args.output/"COMPLETE.json", dict(status="complete", modes=list(MODES)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("clone", "rain", "fog"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--smoke-frames", type=int, default=837)
    parser.add_argument("--worker-mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--backend-profile", choices=("native_vggt",), default="native_vggt")
    args = parser.parse_args(argv)
    args.frames = args.smoke_frames
    if not 3 <= args.frames <= 837:
        parser.error("--smoke-frames must be in [3,837]")
    return args


def main():
    args = parse_args()
    target = args.output/args.worker_mode if args.worker_mode else args.output
    if target.exists():
        raise FileExistsError(target)
    try:
        if args.worker_mode:
            worker(args)
        else:
            execute(args)
    except Exception as error:
        target.mkdir(parents=True, exist_ok=True)
        write_json(target/"FAILED.json", dict(reason=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
