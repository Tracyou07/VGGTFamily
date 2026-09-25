"""Staged v10 ScanNet-50 inference and the existing FastVGGT evaluation protocol.

The public `run` command handles one (scene, frame budget) pair. Model forward,
sparse stitching and GT scoring run in separate processes so their large live
objects cannot accumulate. Incomplete attempts remain for diagnosis; successful
scoring authorizes only allowlisted scratch-file cleanup.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

from experiments.ours_v7.backend_profiles import early_prepare
_EARLY_PROFILE = early_prepare(sys.argv[1:]) if __name__ == "__main__" and "_forward" in sys.argv else None

import numpy as np

from experiments.ours_v10.scannet50_contract import (
    BASELINE_ROOT, BUDGETS, EVAL_ROOT, OUTPUT_BASE, PREPARED_ROOT, SCENE_LIST,
    SCRATCH_BASE, atomic_json, aggregate_completed, audit_frame_selection,
    cleanup_regenerable, read_json, read_scene_list, safe_run_paths, sha256,
    validate_metrics,
)
from experiments.ours_v6.windows import make_windows


CHECKPOINT = Path("/data/yjh/share/pretrained/VGGT-1B/model.safetensors")
CHECKPOINT_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
MODES = ("camera_global_overlap", "overlap_correspondence")
POINT_CONFIDENCE_MIN = 1.0
PYTHON = Path("/home/ubuntu/anaconda3/envs/vggt-gx/bin/python")


def _source_identity():
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if dirty:
        raise RuntimeError("v10 ScanNet-50 requires a committed clean worktree")
    files = [
        root / "experiments/ours_v10/scannet50.py",
        root / "experiments/ours_v10/scannet50_contract.py",
        root / "vggt/v10/model.py", root / "vggt/v10/scheduler.py",
        root / "vggt/v10/attention.py", root / "vggt/v9/sparse_alignment.py",
        EVAL_ROOT / "docs/protocol.md", EVAL_ROOT / "scannet_eval/data.py",
        EVAL_ROOT / "scannet_eval/sens.py", EVAL_ROOT / "scannet_eval/fastvggt_eval.py",
        EVAL_ROOT / "scannet_eval/vendor/fastvggt_eval_utils.py",
    ]
    return dict(commit=head, files={str(path): sha256(path) for path in files})


def _window_schedule(frame_count):
    return make_windows(frame_count, 60, 10)


def _estimate_scratch_bytes(frames):
    replica_frames = sum(hi - lo for lo, hi in _window_schedule(frames))
    input_bytes = frames * 3 * 392 * 518 * 4
    # Local output includes pose, depth, world points and confidence. The
    # allowance is deliberately above the measured fixed-scene local.npz rate.
    return input_bytes + replica_frames * 8_000_000 + 8 * 2**30


def _disk_preflight(scratch, output, frames):
    scratch_parent = Path(scratch)
    output_parent = Path(output)
    while not scratch_parent.exists():
        scratch_parent = scratch_parent.parent
    while not output_parent.exists():
        output_parent = output_parent.parent
    root_free = shutil.disk_usage(scratch_parent).free
    data_free = shutil.disk_usage(output_parent).free
    estimate = _estimate_scratch_bytes(frames)
    if root_free < estimate or data_free < 2 * 2**30:
        raise RuntimeError(f"disk reserve inadequate: scratch free={root_free}, "
                           f"required={estimate}, /data free={data_free}, required=2 GiB")
    return dict(scratch_free_bytes=root_free, output_free_bytes=data_free,
                scratch_required_bytes=estimate, output_required_bytes=2 * 2**30)


def _tree_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in owned scratch: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _next_attempt(scratch, name):
    for number in range(1, 10000):
        path = Path(scratch) / f"{name}_{number:04d}"
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("too many retained stage attempts")


def _run_stage(stage, output, scratch, attempt, gpu=None):
    command = [str(PYTHON), "-u", "-m", "experiments.ours_v10.scannet50",
               stage, "--output", str(output), "--scratch", str(scratch),
               "--attempt", str(attempt)]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if stage == "_forward":
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command.extend(["--gpu", str(gpu), "--backend-profile", "native_vggt"])
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                   env=env, check=True)


def _reusable_input(output):
    path = Path(output) / "input_manifest.json"
    if not path.is_file():
        return False
    record = read_json(path)
    data = Path(record["input_path"])
    return data.is_file() and data.stat().st_size == record["bytes"] and sha256(data) == record["sha256"]


def _reusable_forward(output):
    path = Path(output) / "forward_manifest.json"
    if not path.is_file():
        return False
    record, input_record = read_json(path), read_json(Path(output) / "input_manifest.json")
    if record["input_sha256"] != input_record["sha256"]:
        return False
    if len(record["prediction_files"]) != len(record["windows"]):
        return False
    return all(Path(item["path"]).is_file() and sha256(item["path"]) == item["sha256"]
               for item in record["prediction_files"])


def _reusable_stitch(output):
    path = Path(output) / "stitch_manifest.json"
    if not path.is_file():
        return False
    record = read_json(path)
    if record["forward_manifest_sha256"] != sha256(Path(output) / "forward_manifest.json"):
        return False
    trajectory = Path(output) / "trajectory.npz"
    edges = Path(output) / "alignment_edges.json"
    return (trajectory.is_file() and sha256(trajectory) == record["trajectory_sha256"]
            and edges.is_file() and sha256(edges) == record["alignment_edges_sha256"])


def _reusable_score(output):
    from experiments.ours_v10.scannet50_contract import _validated_final
    try:
        _validated_final(output)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return True


@contextmanager
def _run_lock(output):
    with (Path(output) / "run.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _new_run_manifest(scene_id, budget, mode, gpu, paths, selection, disk,
                      memory_optimized=False):
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("VGGT checkpoint hash changed")
    source = _source_identity()
    return dict(run_id=str(uuid.uuid4()), dataset="ScanNet", scene_id=scene_id,
        frame_budget=budget, actual_frames=selection["actual_frames"],
        frame_ids=selection["frame_ids"],
        windows=_window_schedule(selection["actual_frames"]), mode=mode,
        protocol_id="fastvggt_scannet_evo132", gpu_physical_index=gpu,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA256,
        prepared_root=str(PREPARED_ROOT),
        prepared_manifest_sha256=sha256(PREPARED_ROOT / scene_id / "manifest.json"),
        vggt_star_baseline_result_sha256=sha256(
            BASELINE_ROOT / f"vggt_star_f{budget}_scannet50" / scene_id / "result.json"),
        scene_list=str(SCENE_LIST), scene_list_sha256=sha256(SCENE_LIST),
        source=source, disk_preflight=disk,
        configuration=dict(window_size=60, overlap=10, precision="bf16",
            backend_profile="native_vggt", correspondence_attention_path="native_sdpa",
            query_chunk_size=512, cache_local_kv_dtype=True,
            reuse_image_encoding=False, dense_head_frame_chunk=None,
            memory_optimized=bool(memory_optimized),
            offload_head_features=bool(memory_optimized),
            stream_projected_qkv=bool(memory_optimized),
            alignment_mode="sparse_point_camera_joint", ownership="front window first",
            point_source="transformed point-head world_points from owned frames",
            point_confidence_rule="world_points_conf >= 1.0, finite point coordinates"),
        timing_scope=dict(inference_time_ms="forward including CPU prediction transfer + "
                          "sparse stitching including local.npz read; excludes preprocessing, "
                          "local.npz export and GT evaluation"),
        output=str(paths.output), scratch=str(paths.scratch),
        gt_use="preflight and final scannet_eval scoring only")


def run_one(args):
    scene_ids = read_scene_list(SCENE_LIST)
    if len(scene_ids) != 50 or args.scene not in scene_ids:
        raise ValueError("scene is not in the official ScanNet-50 list")
    selection = audit_frame_selection((args.scene,), (args.frames,), PREPARED_ROOT, BASELINE_ROOT)[0]
    paths = safe_run_paths(args.output_root, args.scratch_root, args.scene, args.frames)
    disk = _disk_preflight(paths.scratch, paths.output, selection["actual_frames"])
    if paths.output.exists() and not args.resume:
        raise FileExistsError(paths.output)
    if paths.scratch.exists() and not args.resume:
        raise FileExistsError(paths.scratch)
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.scratch.mkdir(parents=True, exist_ok=True)
    with _run_lock(paths.output):
        manifest_path = paths.output / "run_manifest.json"
        if manifest_path.is_file():
            run = read_json(manifest_path)
            if (run["scene_id"], run["frame_budget"], run["frame_ids"],
                    run["mode"], run["gpu_physical_index"]) != (
                    args.scene, args.frames, selection["frame_ids"], args.mode, args.gpu):
                raise ValueError("resume request differs from the original run")
            if bool(run.get("configuration", {}).get("memory_optimized", False)) != args.memory_optimized:
                raise ValueError("resume memory optimization differs from the original run")
            if run["source"] != _source_identity() or sha256(CHECKPOINT) != run["checkpoint_sha256"]:
                raise ValueError("resume source or checkpoint changed")
        else:
            run = _new_run_manifest(args.scene, args.frames, args.mode, args.gpu,
                                    paths, selection, disk, args.memory_optimized)
            atomic_json(manifest_path, run)
            atomic_json(paths.scratch / "owner.json",
                        dict(output=str(paths.output.resolve()), run_id=run["run_id"]))
        if read_json(paths.scratch / "owner.json") != dict(
                output=str(paths.output.resolve()), run_id=run["run_id"]):
            raise ValueError("scratch ownership mismatch")
        if (paths.output / "COMPLETE.json").is_file():
            if not _reusable_score(paths.output):
                raise ValueError("COMPLETE marker has incomplete result")
            print(f"already complete: {paths.output}", flush=True)
            return
        try:
            # A crash during allowlisted cleanup must not re-create input or
            # prediction files after scoring has already been confirmed.
            if _reusable_score(paths.output):
                print("resume: verified final scores; finishing cleanup only", flush=True)
                receipt = cleanup_regenerable(paths.scratch, paths.output, run["run_id"])
                atomic_json(paths.output / "COMPLETE.json", dict(status="complete",
                    scene_id=args.scene, frame_budget=args.frames,
                    actual_frames=run["actual_frames"],
                    cleanup_receipt_sha256=sha256(paths.output / "cleanup_receipt.json"),
                    deleted_bytes=receipt["deleted_bytes"]))
                return
            if not _reusable_input(paths.output):
                attempt = _next_attempt(paths.scratch, "input")
                _run_stage("_input", paths.output, paths.scratch, attempt)
            else:
                print("resume: reusing verified input tensor", flush=True)
            _disk_preflight(paths.scratch, paths.output, selection["actual_frames"])
            if not _reusable_forward(paths.output):
                attempt = _next_attempt(paths.scratch, "forward")
                _run_stage("_forward", paths.output, paths.scratch, attempt, args.gpu)
            else:
                print("resume: reusing verified window predictions", flush=True)
            if not _reusable_stitch(paths.output):
                attempt = _next_attempt(paths.scratch, "stitch")
                _run_stage("_stitch", paths.output, paths.scratch, attempt)
            else:
                print("resume: reusing verified sparse alignment", flush=True)
            if not _reusable_score(paths.output):
                attempt = _next_attempt(paths.scratch, "score")
                _run_stage("_score", paths.output, paths.scratch, attempt)
            else:
                print("resume: reusing verified scene scores", flush=True)
            scratch_peak = _tree_bytes(paths.scratch)
            atomic_json(paths.output / "disk_peak.json", dict(
                owned_scratch_peak_before_cleanup_bytes=scratch_peak,
                measurement="sum of owned regular files immediately before cleanup; "
                            "scratch grows monotonically through the stages",
                root_free_before_cleanup_bytes=shutil.disk_usage(paths.scratch).free,
                data_free_before_cleanup_bytes=shutil.disk_usage(paths.output).free))
            receipt = cleanup_regenerable(paths.scratch, paths.output, run["run_id"])
            atomic_json(paths.output / "COMPLETE.json", dict(status="complete",
                scene_id=args.scene, frame_budget=args.frames, actual_frames=run["actual_frames"],
                cleanup_receipt_sha256=sha256(paths.output / "cleanup_receipt.json"),
                deleted_bytes=receipt["deleted_bytes"]))
            print(f"complete: {paths.output}", flush=True)
        except Exception as error:
            atomic_json(paths.output / "FAILED.json", dict(
                reason=str(error), traceback=traceback.format_exc(),
                retained_scratch=str(paths.scratch)))
            raise


def _worker_paths(args):
    output, scratch, attempt = Path(args.output), Path(args.scratch), Path(args.attempt)
    run = read_json(output / "run_manifest.json")
    if attempt.parent != scratch or not attempt.is_dir():
        raise ValueError("stage attempt is outside owned scratch")
    if read_json(scratch / "owner.json") != dict(output=str(output.resolve()), run_id=run["run_id"]):
        raise ValueError("stage scratch ownership mismatch")
    return output, scratch, attempt, run


def stage_input(args):
    from scannet_eval.data import load_scene
    from scannet_eval.fastvggt_eval import validate_scene_inputs
    from vggt.utils.load_fn import load_and_preprocess_images
    import torch
    from experiments.ours_v9.vkitti_predict import tensor_sha256
    output, _, attempt, run = _worker_paths(args)
    prepared_manifest = PREPARED_ROOT / run["scene_id"] / "manifest.json"
    if sha256(prepared_manifest) != run["prepared_manifest_sha256"]:
        raise ValueError("prepared ScanNet scene manifest changed")
    scene = load_scene(PREPARED_ROOT, run["scene_id"], max_frames=run["frame_budget"])
    validate_scene_inputs(scene)
    if list(scene.frame_ids) != run["frame_ids"]:
        raise ValueError("selected original frame IDs changed")
    file_table = read_json(prepared_manifest)["files"]
    source_rows = []
    scene_dir = PREPARED_ROOT / run["scene_id"]
    for path in scene.image_paths:
        relative = path.relative_to(scene_dir).as_posix()
        digest = sha256(path)
        if file_table.get(relative) != digest:
            raise ValueError(f"prepared RGB hash changed: {relative}")
        source_rows.append((relative, digest))
    source_digest = hashlib.sha256(json.dumps(source_rows, separators=(",", ":")).encode()).hexdigest()
    started = time.perf_counter()
    images = load_and_preprocess_images([str(p) for p in scene.image_paths])
    if (tuple(images.shape[:2]) != (len(scene.frame_ids), 3) or
            tuple(images.shape[-2:]) != (392, 518) or images.dtype != torch.float32 or
            not torch.isfinite(images).all()):
        raise ValueError("unexpected preprocessed ScanNet tensor")
    tensor_hash = tensor_sha256(images)
    path = attempt / "inputs.pt"
    with path.open("xb") as stream:
        torch.save(dict(images=images, frame_ids=[f"{i:06d}" for i in scene.frame_ids],
                        preprocessing=dict(loader="vggt.utils.load_fn.load_and_preprocess_images",
                                           shape=list(images.shape), dtype=str(images.dtype))), stream)
    atomic_json(output / "input_manifest.json", dict(input_path=str(path),
        bytes=path.stat().st_size, sha256=sha256(path), tensor_sha256=tensor_hash,
        frame_ids=run["frame_ids"], shape=list(images.shape),
        selected_rgb_sha256=source_digest,
        preprocessing_seconds=time.perf_counter()-started,
        loader="vggt.utils.load_fn.load_and_preprocess_images"))
    print(f"input prepared: {len(scene.frame_ids)} frames, {path.stat().st_size} bytes", flush=True)


@contextmanager
def _heartbeat(label, interval=30):
    stop = threading.Event()
    started = time.perf_counter()
    def report():
        while not stop.wait(interval):
            print(f"{label}: {time.perf_counter()-started:.0f}s elapsed", flush=True)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def stage_forward(args):
    from safetensors.torch import load_file
    import torch
    from experiments.ours_v6.runtime import preflight
    from experiments.ours_v7.backend_profiles import apply_backend_profile, snapshot
    from experiments.ours_v7.diagnostic_export import save_npz_fast
    from experiments.ours_v9.vkitti_predict import prediction_values, tensor_sha256
    from vggt.models.vggt import VGGT
    from vggt.v10.model import WindowReconstructor
    output, _, attempt, run = _worker_paths(args)
    if _EARLY_PROFILE != "native_vggt" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("native backend and physical GPU must be configured before CUDA")
    if args.gpu != run["gpu_physical_index"]:
        raise ValueError("physical GPU differs from run manifest")
    gpu_preflight = preflight(args.gpu, output, min_disk_gib=2)
    apply_backend_profile("native_vggt", torch)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()  # once for the whole forward process
    torch.manual_seed(2026); np.random.seed(2026)
    input_record = read_json(output / "input_manifest.json")
    if sha256(input_record["input_path"]) != input_record["sha256"]:
        raise ValueError("prepared tensor changed")
    saved = torch.load(input_record["input_path"], map_location="cpu", weights_only=True)
    ids = [f"{i:06d}" for i in run["frame_ids"]]
    if saved["frame_ids"] != ids or tensor_sha256(saved["images"]) != input_record["tensor_sha256"]:
        raise ValueError("prepared tensor frame/hash mismatch")
    if sha256(CHECKPOINT) != run["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")
    started = time.perf_counter()
    model = VGGT().eval().requires_grad_(False)
    weights = load_file(str(CHECKPOINT))
    model.load_state_dict(weights, strict=True); del weights
    model.cuda(); torch.cuda.synchronize()
    model_load_seconds = time.perf_counter()-started
    wrapper = WindowReconstructor(model)
    started = time.perf_counter()
    print(f"forward started: {run['scene_id']} f{run['frame_budget']} {run['mode']}", flush=True)
    with _heartbeat("forward"), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = wrapper(saved["images"], ids, mode=run["mode"], window_size=60,
            overlap=10, query_chunk_size=512, reuse_image_encoding=False,
            cache_local_kv_dtype=True, correspondence_attention_path="native_sdpa",
            dense_head_frame_chunk=None,
            offload_head_features=run["configuration"].get("offload_head_features", False),
            stream_projected_qkv=run["configuration"].get("stream_projected_qkv", False))
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter()-started
    if result["windows"] != [tuple(x) for x in run["windows"]]:
        raise ValueError("v10 returned a different window schedule")
    records = []
    export_seconds = 0.
    for index, (prediction, (lo, hi)) in enumerate(zip(result["predictions"], result["windows"])):
        phase = time.perf_counter()
        values = prediction_values(prediction, ids[lo:hi])
        folder = attempt / "windows" / f"{index:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / "local.npz"
        save_npz_fast(path, **values)
        records.append(dict(window=index, lo=lo, hi=hi, path=str(path),
                            bytes=path.stat().st_size, sha256=sha256(path), fields=list(values)))
        export_seconds += time.perf_counter()-phase
        result["predictions"][index] = None
        print(f"exported window {index+1}/{len(result['windows'])}", flush=True)
    torch.cuda.synchronize()
    if len(records) != len(run["windows"]):
        raise ValueError("window export incomplete")
    atomic_json(output / "forward_manifest.json", dict(
        frame_ids=run["frame_ids"], windows=run["windows"],
        input_sha256=input_record["sha256"], input_tensor_sha256=input_record["tensor_sha256"],
        checkpoint_sha256=run["checkpoint_sha256"], prediction_files=records,
        correspondence=result["correspondence"], gpu_uuid=gpu_preflight["gpu_uuid"],
        backend=dict(requested="native_vggt", effective=snapshot(torch)),
        timing=dict(model_load_seconds=model_load_seconds, forward_seconds=forward_seconds,
                    export_seconds=export_seconds, model_stages=result["timing"]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope="fresh GPU process, one reset before model load, through CPU outputs and export",
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print(f"forward complete: {forward_seconds:.3f}s", flush=True)


def _load_local(record, expected_ids):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"frozen local prediction changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    if list(prediction["frame_ids"]) != [f"{i:06d}" for i in expected_ids]:
        raise ValueError("window original frame IDs differ")
    return prediction


def stage_stitch(args):
    from vggt.v9.sparse_alignment import SparseAlignmentConfig, V9AlignmentStitcher
    output, _, attempt, run = _worker_paths(args)
    forward = read_json(output / "forward_manifest.json")
    if forward["frame_ids"] != run["frame_ids"] or forward["windows"] != run["windows"]:
        raise ValueError("frozen forward frame/window manifest mismatch")
    stitcher = V9AlignmentStitcher(attempt / "alignment", SparseAlignmentConfig())
    started = time.perf_counter()
    loading_seconds = 0.
    edges = []
    for index, (record, (lo, hi)) in enumerate(zip(forward["prediction_files"], run["windows"])):
        phase = time.perf_counter()
        prediction = _load_local(record, run["frame_ids"][lo:hi])
        loading_seconds += time.perf_counter()-phase
        fresh, transformed = stitcher.add(prediction, index)
        expected_fresh = list(range(hi-lo)) if index == 0 else list(
            range(run["windows"][index-1][1]-lo, hi-lo))
        if fresh != expected_fresh:
            raise ValueError("front-window ownership changed")
        del transformed
        if index:
            edge = read_json(attempt / "alignment" / f"edge_{index-1:04d}_{index:04d}.json")
            if edge.get("status") != "success" or edge["adjacent"]["scale"] <= 0:
                raise ValueError("invalid sparse alignment edge")
            edges.append(edge)
        print(f"stitched window {index+1}/{len(run['windows'])}", flush=True)
    trajectory = stitcher.finish([f"{i:06d}" for i in run["frame_ids"]])
    stitch_seconds = time.perf_counter()-started
    if len(edges) != len(run["windows"])-1:
        raise ValueError("sparse alignment edge count mismatch")
    np.savez_compressed(attempt / "trajectory.npz", **trajectory)
    atomic_json(attempt / "alignment_edges.json", edges)
    shutil.copy2(attempt / "trajectory.npz", output / "trajectory.npz")
    shutil.copy2(attempt / "alignment_edges.json", output / "alignment_edges.json")
    retained = output / "alignment"
    retained.mkdir(exist_ok=True)
    for path in (attempt / "alignment").glob("*.json"):
        shutil.copy2(path, retained / path.name)
    atomic_json(output / "stitch_manifest.json", dict(
        forward_manifest_sha256=sha256(output / "forward_manifest.json"),
        trajectory_sha256=sha256(output / "trajectory.npz"),
        alignment_edges_sha256=sha256(output / "alignment_edges.json"),
        edge_count=len(edges), fallback_edges=[i for i, e in enumerate(edges) if e.get("fallback")],
        selected_pairs=[e.get("selected_pairs") for e in edges],
        stitch_seconds=stitch_seconds, prediction_loading_seconds=loading_seconds,
        alignment_only_seconds=stitch_seconds-loading_seconds,
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    print(f"sparse stitching complete: {stitch_seconds:.3f}s", flush=True)


def _point_cloud(output, run, forward, trajectory):
    """Use every owned valid point-head pixel; no fitting or GT in this step."""
    from experiments.ours_v3.geometry import Sim3
    points = []
    source_window = trajectory["source_window"].astype(int)
    selected_count = 0
    candidate_count = 0
    for index, (record, (lo, hi)) in enumerate(zip(forward["prediction_files"], run["windows"])):
        prediction = _load_local(record, run["frame_ids"][lo:hi])
        transform_record = read_json(Path(output) / "alignment" / f"window_{index:04d}_transform.json")
        transform = Sim3(transform_record["scale"], np.asarray(transform_record["rotation"]),
                         np.asarray(transform_record["translation"]))
        for local_index, global_index in enumerate(range(lo, hi)):
            if source_window[global_index] != index:
                continue
            cloud = prediction["world_points"][local_index]
            confidence = prediction["world_points_conf"][local_index]
            candidate_count += confidence.size
            mask = (np.isfinite(cloud).all(axis=-1) & np.isfinite(confidence) &
                    (confidence >= POINT_CONFIDENCE_MIN))
            selected = cloud[mask]
            if len(selected):
                transformed = transform.apply(selected).astype(np.float32)
                if not np.isfinite(transformed).all():
                    raise ValueError("nonfinite transformed point cloud")
                points.append(transformed)
                selected_count += len(transformed)
        del prediction
        print(f"point extraction window {index+1}/{len(run['windows'])}", flush=True)
    if not points:
        raise ValueError("no valid v10 point-head points for Chamfer")
    return np.concatenate(points, axis=0), dict(candidate_points=candidate_count,
        selected_points=selected_count, confidence_threshold=POINT_CONFIDENCE_MIN,
        source="front-owned point-head world_points transformed by v9 window Sim(3)")


def stage_score(args):
    from scannet_eval.backends.common import Prediction
    from scannet_eval.data import load_scene
    from scannet_eval.fastvggt_eval import evaluate_prediction, validate_scene_inputs
    output, _, attempt, run = _worker_paths(args)
    if sha256(PREPARED_ROOT / run["scene_id"] / "manifest.json") != run["prepared_manifest_sha256"]:
        raise ValueError("prepared ScanNet scene manifest changed before scoring")
    scene = load_scene(PREPARED_ROOT, run["scene_id"], max_frames=run["frame_budget"])
    validate_scene_inputs(scene)
    if list(scene.frame_ids) != run["frame_ids"]:
        raise ValueError("GT evaluator selection differs from frozen run")
    with np.load(output / "trajectory.npz", allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    if list(trajectory["frame_ids"]) != [f"{i:06d}" for i in run["frame_ids"]]:
        raise ValueError("stitched trajectory frame order mismatch")
    forward = read_json(output / "forward_manifest.json")
    stitched = read_json(output / "stitch_manifest.json")
    points, point_meta = _point_cloud(output, run, forward, trajectory)
    inference_seconds = forward["timing"]["forward_seconds"] + stitched["stitch_seconds"]
    prediction = Prediction(points=points, poses_c2w=trajectory["c2w"],
        frame_ids=tuple(run["frame_ids"]), inference_seconds=inference_seconds,
        peak_allocated_bytes=forward["peak_allocated_bytes"],
        peak_reserved_bytes=forward["peak_reserved_bytes"],
        metadata=dict(model="ours_v10", mode=run["mode"], point_cloud=point_meta,
                      timing_scope=run["timing_scope"],
                      no_gt_in_forward_or_sparse_alignment=True))
    started = time.perf_counter()
    print(f"protocol scoring started: {len(points)} predicted points", flush=True)
    with _heartbeat("ScanNet protocol scoring"):
        metrics = validate_metrics(evaluate_prediction(scene, prediction, attempt,
                                                        chamfer_max_dist=.5, plot=False))
    scoring_seconds = time.perf_counter()-started
    if validate_metrics(read_json(attempt / "metrics.json")) != metrics:
        raise ValueError("evaluator metrics file differs from returned scores")
    record = dict(schema_version=1, scene_id=run["scene_id"],
        protocol_id=run["protocol_id"], frame_budget=run["frame_budget"],
        actual_frames=run["actual_frames"], frame_ids=run["frame_ids"],
        metrics=metrics, prediction=dict(inference_seconds=inference_seconds,
            peak_allocated_bytes=forward["peak_allocated_bytes"],
            peak_reserved_bytes=forward["peak_reserved_bytes"],
            metadata=prediction.metadata),
        source_commit=run["source"]["commit"], checkpoint_sha256=run["checkpoint_sha256"],
        input_tensor_sha256=read_json(output / "input_manifest.json")["tensor_sha256"],
        forward_seconds=forward["timing"]["forward_seconds"],
        stitch_seconds=stitched["stitch_seconds"],
        forward_plus_stitch_seconds=inference_seconds,
        export_seconds=forward["timing"]["export_seconds"],
        scoring_seconds=scoring_seconds,
        gpu_uuid=forward["gpu_uuid"],
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "result.json", record)
    atomic_json(output / "summary.json", dict(complete=True, scene_id=run["scene_id"],
        frame_budget=run["frame_budget"], protocol_id=run["protocol_id"],
        metrics_sha256=sha256(output / "metrics.json"),
        result_sha256=sha256(output / "result.json"),
        trajectory_sha256=sha256(output / "trajectory.npz"),
        alignment_edges_sha256=sha256(output / "alignment_edges.json")))
    print(f"protocol scoring complete: Chamfer={metrics['chamfer_distance']:.6f} m", flush=True)


def run_audit(args):
    scenes = read_scene_list(SCENE_LIST)
    if len(scenes) != 50:
        raise ValueError("official ScanNet list is not 50 scenes")
    rows = audit_frame_selection(scenes, BUDGETS)
    result = dict(protocol_id="fastvggt_scannet_evo132", scene_count=len(scenes),
                  budgets=list(BUDGETS), comparisons=len(rows), rows=rows)
    if args.output:
        atomic_json(args.output, result)
    print(f"frame selection verified: {len(scenes)} scenes x {len(BUDGETS)} budgets", flush=True)


def run_aggregate(args):
    scenes = read_scene_list(SCENE_LIST)
    summary = aggregate_completed(args.output_root, args.frames, scenes)
    baselines = {}
    for name in ("vggt_original", "vggt_star", "fastvggt", "streamvggt",
                 "long", "slam", "omega"):
        path = BASELINE_ROOT / f"{name}_f{args.frames}_scannet50" / "summary.json"
        if not path.is_file():
            baselines[name] = dict(status="missing", metrics=None)
            continue
        record = read_json(path)
        baselines[name] = (dict(status="complete", metrics=record.get("average_metrics"),
                                source=str(path), source_sha256=sha256(path))
                           if record.get("complete") is True else
                           dict(status="incomplete", metrics=None))
    summary["baseline_availability"] = baselines
    atomic_json(Path(args.output_root) / f"f{args.frames}" / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("complete", "success_count", "expected_count")}), flush=True)
    if not summary["complete"]:
        raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="one scene and one frame budget")
    run.add_argument("--scene", required=True)
    run.add_argument("--frames", type=int, choices=BUDGETS, required=True)
    run.add_argument("--mode", choices=MODES, default="camera_global_overlap")
    run.add_argument("--gpu", type=int, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--scratch-root", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--memory-optimized", action="store_true",
                     help="offload cached head features and stream projected QKV")
    audit = sub.add_parser("audit-selection")
    audit.add_argument("--output", type=Path)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--frames", type=int, choices=BUDGETS, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    for name in ("_input", "_forward", "_stitch", "_score"):
        stage = sub.add_parser(name)
        stage.add_argument("--output", type=Path, required=True)
        stage.add_argument("--scratch", type=Path, required=True)
        stage.add_argument("--attempt", type=Path, required=True)
        if name == "_forward":
            stage.add_argument("--gpu", type=int, required=True)
            stage.add_argument("--backend-profile", choices=("native_vggt",), required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.command == "run":
        run_one(args)
    elif args.command == "audit-selection":
        run_audit(args)
    elif args.command == "aggregate":
        run_aggregate(args)
    else:
        {"_input": stage_input, "_forward": stage_forward,
         "_stitch": stage_stitch, "_score": stage_score}[args.command](args)


if __name__ == "__main__":
    main()
