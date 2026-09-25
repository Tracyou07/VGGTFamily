"""Predict v10 windows from the frozen ScanNet scene0000_00 CPU tensor.

This entry is separate from the Virtual KITTI Scene20 entry. It produces only
local window predictions; GT is validated before launch and never passed to
the model, alignment, selection, or fallback code.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from experiments.ours_v7.backend_profiles import early_prepare
_EARLY_PROFILE = early_prepare(sys.argv[1:]) if __name__ == "__main__" else None

import numpy as np
import torch

from experiments.ours_v6.runtime import preflight, sha256, source_identity, write_json
from experiments.ours_v6.windows import make_windows
from experiments.ours_v7.diagnostic_export import save_npz_fast
from experiments.ours_v9.vkitti_predict import prediction_values, tensor_sha256


FRAME_COUNTS = (100, 300, 500, 1000)
MODES = ("overlap_correspondence", "camera_global_overlap")
FROZEN_ROOT = Path("/home/ubuntu/yjh/feedforwardreconstruct/ours_v8_experiments/"
                   "20260923T074952Z_scene0000_00_f1000_w60_o10")
FROZEN_INPUT = FROZEN_ROOT / "inputs.pt"
FROZEN_MANIFEST = FROZEN_ROOT / "independent" / "run_manifest.json"
INPUT_SHA256 = "0c98205a8acef8558fbf522de364df7bd0a9e482f0b27b9ca54a08690085610b"
CHECKPOINT_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"


def expected_windows(frames):
    if frames not in FRAME_COUNTS:
        raise ValueError(f"unsupported fixed frame count: {frames}")
    windows = make_windows(frames, 60, 10)
    if len(windows) != {100: 2, 300: 6, 500: 10, 1000: 20}[frames]:
        raise ValueError("unexpected window count")
    return windows


def validate_gt_poses(scene_root, frame_ids):
    """Read GT for preflight only; return no poses to the forward path."""
    pose_dir = Path(scene_root) / "pose"
    for frame in frame_ids:
        path = pose_dir / f"{frame}.txt"
        if not path.is_file():
            raise ValueError(f"missing ScanNet GT pose: {path}")
        pose = np.loadtxt(path)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5) or
                np.linalg.det(pose[:3, :3]) <= 0):
            raise ValueError(f"invalid ScanNet GT pose: {path}")


def validate_frozen_input(path, source, frames, *, shape=(1000, 3, 392, 518), check_gt=True):
    windows = expected_windows(frames)
    path = Path(path).resolve()
    if path != Path(source["configuration"]["input"]).resolve():
        raise ValueError("frozen input path differs from source manifest")
    if sha256(path) != source["input_sha256"]:
        raise ValueError("frozen input file hash changed")
    checkpoint = Path(source["checkpoint"])
    if sha256(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("checkpoint hash changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    original_ids = [f"{i:06d}" for i in range(1000)]
    if list(source["frame_ids"]) != original_ids or list(saved["frame_ids"]) != original_ids:
        raise ValueError("frozen frame ID order changed")
    if tuple(saved["images"].shape) != tuple(shape) or saved["images"].dtype != torch.float32:
        raise ValueError("frozen image tensor shape or dtype changed")
    if list(saved["preprocessing"]["shape"]) != list(shape):
        raise ValueError("frozen preprocessing metadata changed")
    if check_gt:
        scene_root = Path(saved["scene_root"]).resolve()
        if scene_root.name != "scene0000_00":
            raise ValueError("unexpected ScanNet scene root")
        validate_gt_poses(scene_root, original_ids[:frames])
    ids = original_ids[:frames]
    prefix_hash = tensor_sha256(saved["images"][:frames])
    return saved, source, ids, prefix_hash, windows


def execute(args):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.v10.model import WindowReconstructor
    from experiments.ours_v7.backend_profiles import apply_backend_profile, snapshot

    if _EARLY_PROFILE != "native_vggt":
        raise ValueError("native_vggt profile must be prepared before CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must match physical --gpu")
    if args.frozen_input.resolve() != FROZEN_INPUT.resolve():
        raise ValueError("only the pinned frozen ScanNet inputs.pt is supported")
    if args.source_manifest.resolve() != FROZEN_MANIFEST.resolve():
        raise ValueError("only the pinned frozen v8 source manifest is supported")
    target = args.output_root / args.mode
    if target.exists():
        raise FileExistsError(target)
    preflight_result = preflight(args.gpu, args.output_root, min_disk_gib=20)
    apply_backend_profile("native_vggt", torch)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()  # once in this fresh process
    torch.manual_seed(2026)
    np.random.seed(2026)

    read_start = time.perf_counter()
    source = json.loads(args.source_manifest.read_text())
    if source["input_sha256"] != INPUT_SHA256 or source["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("pinned source manifest input/checkpoint hash changed")
    saved, source, ids, prefix_hash, windows = validate_frozen_input(
        args.frozen_input, source, args.frames)
    images = saved["images"][:args.frames]
    input_read_seconds = time.perf_counter() - read_start
    identity = source_identity()
    if identity["status"]:
        raise RuntimeError("v10 prediction requires a committed clean worktree")

    args.output_root.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    model_start = time.perf_counter()
    model = VGGT().eval().requires_grad_(False)
    weights = load_file(source["checkpoint"])
    model.load_state_dict(weights, strict=True)
    del weights
    model.cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_start
    wrapper = WindowReconstructor(model)

    forward_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = wrapper(images, ids, mode=args.mode, window_size=60, overlap=10,
            query_chunk_size=512, reuse_image_encoding=False,
            cache_local_kv_dtype=True, correspondence_attention_path="native_sdpa",
            dense_head_frame_chunk=None)
    torch.cuda.synchronize()
    if result["windows"] != windows or len(result["predictions"]) != len(windows):
        raise ValueError("v10 returned the wrong window schedule")
    forward_seconds = time.perf_counter() - forward_start

    prediction_files = []
    export_seconds = 0.0
    for index, (prediction, (lo, hi)) in enumerate(zip(result["predictions"], windows)):
        started = time.perf_counter()
        values = prediction_values(prediction, ids[lo:hi])
        folder = target / "windows" / f"{index:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        output = folder / "local.npz"
        save_npz_fast(output, **values)
        export_seconds += time.perf_counter() - started
        prediction_files.append(dict(window=index, lo=lo, hi=hi,
            path=str(output), sha256=sha256(output), bytes=output.stat().st_size,
            fields=list(values)))
    torch.cuda.synchronize()
    if len(prediction_files) != len(windows):
        raise ValueError("prediction export incomplete")

    manifest = dict(dataset="ScanNet", scene="scene0000_00",
        communication_mode=args.mode, frame_ids=ids, windows=windows,
        sources=identity, frozen_source_manifest=str(args.source_manifest),
        frozen_source_manifest_sha256=sha256(args.source_manifest),
        preprocessing=saved["preprocessing"], input_sha256=source["input_sha256"],
        image_tensor_sha256=prefix_hash,
        checkpoint=source["checkpoint"], checkpoint_sha256=source["checkpoint_sha256"],
        precision="bf16", configuration=dict(input=str(args.frozen_input),
            frames=args.frames, window_size=60, overlap=10, backend_profile="native_vggt",
            correspondence_attention_path="native_sdpa", query_chunk_size=512,
            cache_local_kv_dtype=True, dense_head_frame_chunk=None,
            reuse_image_encoding=False, alignment_mode="sparse_point_camera_joint",
            ownership="front window first"),
        backend_profile=dict(requested="native_vggt", effective=snapshot(torch)),
        preflight=preflight_result, gpu_uuid=preflight_result["gpu_uuid"],
        prediction_files=prediction_files, correspondence=result["correspondence"],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope="fresh process from CUDA initialization through local.npz export; no phase reset",
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        timing=dict(model_load_seconds=model_load_seconds,
            input_read_seconds=input_read_seconds, forward_seconds=forward_seconds,
            export_seconds=export_seconds),
        gt_validated_preflight_only=True, no_alignment_or_gt_evaluation=True,
        no_point_cloud_export=True)
    write_json(target / "run_manifest.json", manifest)
    write_json(target / "COMPLETE.json", dict(status="complete", mode=args.mode))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input", type=Path, default=FROZEN_INPUT)
    parser.add_argument("--source-manifest", type=Path, default=FROZEN_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--frames", type=int, choices=FRAME_COUNTS, required=True)
    parser.add_argument("--backend-profile", choices=("native_vggt",), required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    target = args.output_root / args.mode
    if target.exists():
        raise FileExistsError(target)
    try:
        execute(args)
    except Exception as error:
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "FAILED.json", dict(reason=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
