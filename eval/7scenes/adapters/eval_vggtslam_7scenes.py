"""Headless VGGT-SLAM evaluation under the FastVGGT 7-Scenes protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3]
SLAM_ROOT = ROOT / "vggtslam"
BASE_MODELS = ROOT / "vggtlong" / "base_models"
ADAPTER_DIR = Path(__file__).resolve().parent

if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from registered_data import DEFAULT_DATA_ROOT, prepare_evaluation


DEFAULT_RAW_DATA_ROOT = "/data/yjh/share/datasets/7scenes"


def resolve_vggt_source() -> Path:
    """Use the VGGT-SPARK checkout when it is complete, otherwise fall back."""

    official = SLAM_ROOT / "third_party" / "vggt"
    if (official / "vggt" / "models" / "vggt.py").is_file():
        return official
    return BASE_MODELS


sys.path.insert(0, str(SLAM_ROOT / "third_party" / "salad"))
sys.path.insert(0, str(SLAM_ROOT))
sys.path.insert(0, str(resolve_vggt_source()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kf", type=int, required=True, choices=[3, 10])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=DEFAULT_RAW_DATA_ROOT)
    p.add_argument("--registered-depth-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf-threshold", type=float, default=25.0)
    p.add_argument("--submap-size", type=int, default=16)
    p.add_argument("--max-loops", type=int, default=1, choices=[0, 1])
    p.add_argument("--lc-thres", type=float, default=0.95)
    p.add_argument("--max-points", type=int, default=999999)
    p.add_argument(
        "--sequence",
        action="append",
        help="Optional scene/sequence filter; repeat for more than one sequence.",
    )
    return p.parse_args()


class NullViewer:
    """Avoid starting a Viser server during a batch evaluation."""

    def __init__(self, *args, **kwargs):
        pass


class NoOpImageRetrieval:
    """Explicit no-loop ablation that avoids loading SALAD weights."""

    def get_all_submap_embeddings(self, submap):
        return np.zeros((len(submap.get_all_frames()), 1), dtype=np.float32)

    def find_loop_closures(self, *args, **kwargs):
        return []


def install_noop_salad_import():
    """Let the headless no-loop protocol import Solver without SALAD extras."""

    import types

    salad_pkg = types.ModuleType("salad")
    salad_eval = types.ModuleType("salad.eval")
    salad_eval.load_model = lambda *args, **kwargs: None
    salad_pkg.eval = salad_eval
    sys.modules["salad"] = salad_pkg
    sys.modules["salad.eval"] = salad_eval


@contextmanager
def offline_dinov2_hub():
    """Route SALAD's DINOv2 hub request to the shared local checkout."""

    original_load = torch.hub.load
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not local_repo.is_dir():
        raise FileNotFoundError(f"Missing local DINOv2 hub checkout: {local_repo}")

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs["source"] = "local"
            repo_or_dir = str(local_repo)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original_load


def load_model(checkpoint: str, device: str):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT as BaseVGGT

    model = BaseVGGT()
    if checkpoint.endswith(".safetensors"):
        state = load_file(checkpoint, device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    del state
    model.eval().to(device)

    class AutocastModel(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.total_forward_seconds = 0.0
            self.forward_calls = 0

        def forward(self, images, *args, **kwargs):
            if images.is_cuda:
                torch.cuda.synchronize(images.device)
            start = time.perf_counter()
            device_type = "cuda" if images.is_cuda else "cpu"
            dtype = torch.bfloat16 if images.is_cuda else torch.float32
            with torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=images.is_cuda,
            ):
                output = self.wrapped(images, *args, **kwargs)
            if images.is_cuda:
                torch.cuda.synchronize(images.device)
            self.total_forward_seconds += time.perf_counter() - start
            self.forward_calls += 1
            return output

    return AutocastModel(model).eval()


def test_sequences(data_root: Path, only=None):
    only = set(only) if only else None
    for scene_dir in sorted(data_root.iterdir()):
        split = scene_dir / "TestSplit.txt"
        if not split.is_file():
            continue
        for line in split.read_text().splitlines():
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                seq = f"seq-{digits.zfill(2)}"
                if only is None or f"{scene_dir.name}/{seq}" in only:
                    yield scene_dir.name, seq


def sorted_color_paths(seq_dir: Path):
    return sorted(
        seq_dir.glob("frame-*.color.png"),
        key=lambda p: int(p.stem.split("-")[1].split(".")[0]),
    )


def uniformly_sample_frames(frame_paths, kf: int):
    if kf <= 0:
        raise ValueError("kf must be positive")
    return list(frame_paths)[::kf]


def iter_submap_windows(frame_paths, submap_size: int):
    if submap_size <= 0:
        raise ValueError("submap_size must be positive")
    frame_paths = list(frame_paths)
    for start in range(0, len(frame_paths), submap_size):
        window = frame_paths[start : start + submap_size + 1]
        if start > 0 and len(window) == 1:
            break
        if window:
            yield window


def gt_points_for_frames(frame_paths, registered_depth_dir=None):
    frame_paths = list(frame_paths)
    if not frame_paths:
        return np.empty((0, 3), dtype=np.float32)

    points = []
    registered_depth_dir = (
        Path(registered_depth_dir) if registered_depth_dir is not None else None
    )
    K = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    first_stem = frame_paths[0].name.replace(".color.png", "")
    first_pose = np.loadtxt(
        frame_paths[0].with_name(first_stem + ".pose.txt")
    ).astype(np.float64)
    world_to_first = np.linalg.inv(first_pose)
    for color_path in frame_paths:
        stem = color_path.name.replace(".color.png", "")
        depth_path = (
            registered_depth_dir / (stem + ".depth.proj.png")
            if registered_depth_dir is not None
            else color_path.with_name(stem + ".depth.proj.png")
        )
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing registered depth: {depth_path}")
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Unable to read registered depth: {depth_path}")
        pose = np.loadtxt(color_path.with_name(stem + ".pose.txt")).astype(np.float64)
        depth = depth.astype(np.float32) / 1000.0
        valid = np.isfinite(depth) & (depth > 1e-3) & (depth < 10.0)
        yy, xx = np.where(valid)
        z = depth[yy, xx]
        xyz = np.stack(((xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z), axis=1)
        world = (pose[:3, :3] @ xyz.T).T + pose[:3, 3]
        first_camera = (
            world_to_first[:3, :3] @ world.T
        ).T + world_to_first[:3, 3]
        points.append(first_camera)
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32)


def fastvggt_scale_shift_align(pred: np.ndarray, gt: np.ndarray):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1:] != (3,) or gt.shape[1:] != (3,):
        raise ValueError("pred and gt must both have shape (N, 3)")
    if len(pred) == 0 or len(gt) == 0:
        raise ValueError("pred and gt must be non-empty")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("pred and gt must contain only finite points")

    pred_aligned = pred.copy()
    gt_aligned = gt.copy()
    pred_shift_z = float(np.median(pred_aligned[:, 2]))
    gt_shift_z = float(np.median(gt_aligned[:, 2]))
    pred_aligned[:, 2] -= pred_shift_z
    gt_aligned[:, 2] -= gt_shift_z

    pred_center = np.median(pred_aligned, axis=0)
    gt_center = np.median(gt_aligned, axis=0)
    pred_scale = float(np.median(np.linalg.norm(pred_aligned - pred_center, axis=1)))
    gt_scale = float(np.median(np.linalg.norm(gt_aligned - gt_center, axis=1)))
    if pred_scale < 1e-8 or gt_scale < 1e-8:
        raise ValueError("pred and gt must have non-degenerate scale")
    scale = gt_scale / pred_scale
    pred_aligned *= scale
    return pred_aligned, gt_aligned, {
        "scale": float(scale),
        "pred_scale": pred_scale,
        "gt_scale": gt_scale,
        "pred_shift_z": pred_shift_z,
        "gt_shift_z": gt_shift_z,
    }


def estimate_sim3(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1:] != (3,):
        raise ValueError("src and dst must have the same shape (N, 3)")
    if len(src) < 3 or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("src and dst need at least three finite points")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_variance < 1e-12:
        raise ValueError("source trajectory is degenerate")

    covariance = dst_centered.T @ src_centered / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection[-1, -1] = -1
    rotation = u @ reflection @ vt
    scale = float(
        np.sum(singular_values * np.diag(reflection)) / src_variance
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("estimated Sim(3) scale must be positive and finite")
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_sim3(points, scale, rotation, translation):
    points = np.asarray(points, dtype=np.float64)
    return scale * (points @ np.asarray(rotation).T) + np.asarray(translation)


def camera_center_correspondences(solver, frame_paths):
    estimated_by_frame = {}
    for submap in solver.map.ordered_submaps_by_key():
        if getattr(submap, "get_lc_status", lambda: False)():
            continue
        poses = submap.get_all_poses_world(solver.graph)
        for frame_id, pose in zip(submap.get_frame_ids(), poses):
            estimated_by_frame.setdefault(
                int(round(frame_id)), np.asarray(pose[:3, 3], dtype=np.float64)
            )

    estimated = []
    ground_truth = []
    matched_frame_ids = []
    for color_path in frame_paths:
        frame_id = int(color_path.stem.split("-")[1].split(".")[0])
        if frame_id not in estimated_by_frame:
            continue
        stem = color_path.name.replace(".color.png", "")
        pose = np.loadtxt(color_path.with_name(stem + ".pose.txt"))
        estimated.append(estimated_by_frame[frame_id])
        ground_truth.append(np.asarray(pose[:3, 3], dtype=np.float64))
        matched_frame_ids.append(frame_id)

    if not estimated:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            [],
        )
    return np.stack(estimated), np.stack(ground_truth), matched_frame_ids


def trajectory_diagnostics(solver, frame_paths):
    estimated, ground_truth, frame_ids = camera_center_correspondences(
        solver, frame_paths
    )
    scale, rotation, translation = estimate_sim3(estimated, ground_truth)
    aligned = apply_sim3(estimated, scale, rotation, translation)
    raw_ate = float(
        np.sqrt(np.mean(np.sum((estimated - ground_truth) ** 2, axis=1)))
    )
    aligned_ate = float(
        np.sqrt(np.mean(np.sum((aligned - ground_truth) ** 2, axis=1)))
    )
    return {
        "matched_poses": len(frame_ids),
        "raw_ate_m": raw_ate,
        "sim3_ate_m": aligned_ate,
        "sim3_scale": float(scale),
        "sim3_rotation": rotation.tolist(),
        "sim3_translation": translation.tolist(),
    }


def sample_points(points: np.ndarray, max_points: int, seed: int):
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), max_points, replace=False)]


def cloud_metrics(pred: np.ndarray, gt: np.ndarray, max_points: int, seed: int):
    pred = sample_points(pred[np.isfinite(pred).all(axis=1)], max_points, seed)
    gt = sample_points(gt[np.isfinite(gt).all(axis=1)], max_points, seed + 1)
    if len(pred) < 20 or len(gt) < 20:
        raise RuntimeError(f"insufficient points: pred={len(pred)} gt={len(gt)}")

    import open3d as o3d

    pred_pcd = o3d.geometry.PointCloud()
    gt_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(pred)
    gt_pcd.points = o3d.utility.Vector3dVector(gt)
    reg = o3d.pipelines.registration.registration_icp(
        pred_pcd,
        gt_pcd,
        0.1,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_pcd.transform(reg.transformation)
    pred = np.asarray(pred_pcd.points)
    gt = np.asarray(gt_pcd.points)
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    acc = float(gt_tree.query(pred, workers=-1)[0].mean())
    comp = float(pred_tree.query(gt, workers=-1)[0].mean())

    normal_search = o3d.geometry.KDTreeSearchParamKNN(knn=min(30, len(pred) - 1))
    pred_pcd.estimate_normals(normal_search)
    gt_pcd.estimate_normals(normal_search)
    pred_normals = np.asarray(pred_pcd.normals)
    gt_normals = np.asarray(gt_pcd.normals)
    pred_to_gt = gt_tree.query(pred, workers=-1)[1]
    gt_to_pred = pred_tree.query(gt, workers=-1)[1]
    nc1 = float(
        np.abs(np.sum(pred_normals * gt_normals[pred_to_gt], axis=1)).mean()
    )
    nc2 = float(
        np.abs(np.sum(gt_normals * pred_normals[gt_to_pred], axis=1)).mean()
    )
    return {
        "acc": acc,
        "comp": comp,
        "nc1": nc1,
        "nc2": nc2,
        "mean_nc": (nc1 + nc2) / 2.0,
    }


def aligned_cloud_metrics(
    pred: np.ndarray, gt: np.ndarray, max_points: int, seed: int
):
    pred = pred[np.isfinite(pred).all(axis=1)]
    gt = gt[np.isfinite(gt).all(axis=1)]
    aligned_pred, aligned_gt, metadata = fastvggt_scale_shift_align(pred, gt)
    return cloud_metrics(aligned_pred, aligned_gt, max_points, seed), metadata


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    registered_depth_root = Path(args.registered_depth_root)
    output_dir = Path(args.output_dir)
    registration = prepare_evaluation(
        registered_depth_root,
        output_dir,
        kf=args.kf,
        source_data_root=data_root,
    )

    if args.max_loops == 0:
        install_noop_salad_import()
    import vggt_slam.solver as solver_module

    solver_module.Viewer = NullViewer
    if args.max_loops == 0:
        solver_module.ImageRetrieval = NoOpImageRetrieval
    else:
        with offline_dinov2_hub():
            shared_image_retrieval = solver_module.ImageRetrieval()
        solver_module.ImageRetrieval = lambda: shared_image_retrieval
    from vggt_slam.solver import Solver

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    records = []

    sequences = list(test_sequences(data_root, args.sequence))
    if args.sequence and len(sequences) != len(set(args.sequence)):
        found = {f"{scene}/{seq}" for scene, seq in sequences}
        missing = sorted(set(args.sequence) - found)
        raise ValueError(f"Requested sequences not found in test split: {missing}")

    for scene, seq in sequences:
        seq_dir = data_root / scene / seq
        registered_depth_dir = registered_depth_root / scene / seq
        color_paths = sorted_color_paths(seq_dir)
        if not color_paths:
            raise FileNotFoundError(f"No color frames found in {seq_dir}")
        selected = uniformly_sample_frames(color_paths, args.kf)
        solver = Solver(
            init_conf_threshold=args.conf_threshold,
            lc_thres=args.lc_thres,
        )
        forward_start = model.total_forward_seconds
        forward_calls_start = model.forward_calls
        pipeline_start = time.perf_counter()
        with torch.no_grad():
            for window in iter_submap_windows(selected, args.submap_size):
                predictions = solver.run_predictions(
                    [str(path) for path in window],
                    model,
                    args.max_loops,
                    None,
                    None,
                )
                solver.add_points(predictions)
                solver.graph.optimize()

        pred_parts = [
            submap.get_points_in_world_frame(solver.graph)
            for submap in solver.map.ordered_submaps_by_key()
            if not submap.get_lc_status()
        ]
        pred = np.concatenate(pred_parts, axis=0) if pred_parts else np.empty((0, 3), dtype=np.float32)
        pipeline_time_ms = (time.perf_counter() - pipeline_start) * 1000.0
        inference_time_ms = (
            model.total_forward_seconds - forward_start
        ) * 1000.0
        inference_calls = model.forward_calls - forward_calls_start
        gt = gt_points_for_frames(
            selected, registered_depth_dir=registered_depth_dir
        )
        metrics, alignment = aligned_cloud_metrics(
            pred, gt, args.max_points, len(records)
        )
        trajectory = trajectory_diagnostics(solver, selected)
        record = {
            **metrics,
            "scene_id": f"{scene}/{seq}",
            "selected_frames": len(selected),
            "total_frames": len(color_paths),
            "expected_selected_frames": (len(color_paths) + args.kf - 1) // args.kf,
            "pred_points": len(pred),
            "gt_points": len(gt),
            "time_ms": inference_time_ms,
            "inference_time_ms": inference_time_ms,
            "inference_calls": inference_calls,
            "pipeline_time_ms": pipeline_time_ms,
            "loop_closures": solver.graph.get_num_loops(),
            "point_alignment": alignment,
            "trajectory": trajectory,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

        seq_out = output_dir / f"{scene}_{seq}"
        seq_out.mkdir(parents=True, exist_ok=True)
        (seq_out / "selected_frames.txt").write_text("\n".join(str(x) for x in selected) + "\n")
        (seq_out / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        del solver
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    def mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "kf": args.kf,
        "submap_size": args.submap_size,
        "max_loops": args.max_loops,
        "lc_thres": args.lc_thres,
        "valid_sequences": len(records),
        "acc": mean("acc"),
        "comp": mean("comp"),
        "nc1": mean("nc1"),
        "nc2": mean("nc2"),
        "mean_nc": mean("mean_nc"),
        "mean_time_ms": mean("time_ms"),
        "mean_pipeline_time_ms": mean("pipeline_time_ms"),
        "total_loop_closures": int(sum(r["loop_closures"] for r in records)),
        "mean_trajectory_sim3_ate_m": float(
            np.mean([r["trajectory"]["sim3_ate_m"] for r in records])
        ) if records else None,
        "peak_vram_mib": float(torch.cuda.max_memory_reserved() / (1024**2)) if device.startswith("cuda") else None,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "registration": registration,
        "protocol_note": (
            "Uniform every-kf frames read directly from the raw dataset; registered "
            "projected depth read from its generated auxiliary root; GT transformed "
            "to first-camera coordinates; FastVGGT-equivalent median-z shift and "
            "median-radius scale normalization; 0.1 m point-to-point ICP. "
            "Trajectory Sim(3) is diagnostic only."
        ),
    }
    (output_dir / "sequences.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
