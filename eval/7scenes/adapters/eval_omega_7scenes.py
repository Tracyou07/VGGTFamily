"""Unified 7-Scenes evaluator for VGGT-Omega."""

import argparse
from registered_data import DEFAULT_DATA_ROOT, prepare_evaluation
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def center_crop(array, size=224):
    height, width = array.shape[:2]
    top = (height - size) // 2
    left = (width - size) // 2
    return array[top : top + size, left : left + size]


def summarize_metrics(rows):
    if not rows:
        raise ValueError("no valid sequence metrics")
    result = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("acc", "comp", "nc1", "nc2")
    }
    result["mean_nc"] = (result["nc1"] + result["nc2"]) / 2.0
    result["valid_sequences"] = len(rows)
    return result


def evaluate_pointcloud_pair(pred_points, gt_points, icp_threshold=0.1):
    pred_points = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    gt_points = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    if len(pred_points) == 0 or len(gt_points) == 0:
        raise ValueError("point clouds must be non-empty")

    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_points))
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
    registration = o3d.pipelines.registration.registration_icp(
        pred_cloud,
        gt_cloud,
        icp_threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_cloud.transform(registration.transformation)
    pred_cloud.estimate_normals()
    gt_cloud.estimate_normals()

    pred_aligned = np.asarray(pred_cloud.points)
    gt_array = np.asarray(gt_cloud.points)
    pred_normals = np.asarray(pred_cloud.normals)
    gt_normals = np.asarray(gt_cloud.normals)

    gt_tree = cKDTree(gt_array)
    acc_distances, acc_indices = gt_tree.query(pred_aligned, workers=-1)
    pred_tree = cKDTree(pred_aligned)
    comp_distances, comp_indices = pred_tree.query(gt_array, workers=-1)

    nc1 = np.abs(np.sum(gt_normals[acc_indices] * pred_normals, axis=-1))
    nc2 = np.abs(np.sum(pred_normals[comp_indices] * gt_normals, axis=-1))
    return {
        "acc": float(np.mean(acc_distances)),
        "comp": float(np.mean(comp_distances)),
        "nc1": float(np.mean(nc1)),
        "nc2": float(np.mean(nc2)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VGGT-Omega with the FastVGGT 7-Scenes protocol"
    )
    parser.add_argument("--kf", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-points", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare_evaluation(args.data_root, args.output_dir, kf=args.kf)

    adapter_dir = Path(__file__).resolve().parent
    project_root = adapter_dir.parents[2]
    omega_root = project_root / "vggtomega"
    fast_root = project_root / "eval" / "7scenes" / "reference" / "FastVGGT-main"
    sys.path[:0] = [str(omega_root), str(fast_root / "eval"), str(fast_root)]

    import torch
    from torch.utils.data._utils.collate import default_collate

    from criterion import L21, Regr3D_t_ScaleShiftInv
    from data import SevenScenes
    from run import load_model, run_inference

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_log = output_dir / "sequences.jsonl"
    summary_path = output_dir / "summary.json"

    dataset = SevenScenes(
        split="test",
        ROOT=args.data_root,
        resolution=(592, 448),
        num_seq=1,
        full_video=True,
        kf_every=args.kf,
    )
    model = load_model(args.checkpoint, args.device)
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    rng = np.random.default_rng(args.seed)
    torch.cuda.reset_peak_memory_stats()
    rows = []

    with sequence_log.open("w", encoding="utf-8") as log_handle:
        for data_idx in range(len(dataset)):
            batch = default_collate([dataset[data_idx]])
            image_paths = [view["instance"][0] for view in batch]

            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            predictions = run_inference(
                model,
                image_paths,
                image_resolution=512,
                mode="balanced",
                patch_size=16,
                device=args.device,
            )
            torch.cuda.synchronize()
            inference_time_ms = (time.perf_counter() - infer_start) * 1000.0

            predicted_world = predictions["world_points_from_depth"]
            predicted_views = [
                {
                    "pts3d_in_other_view": torch.from_numpy(predicted_world[j])[None],
                }
                for j in range(len(batch))
            ]
            gt_points, pred_points, _, _, masks, _ = criterion.get_all_pts3d_t(
                batch, predicted_views
            )

            pred_parts = []
            gt_parts = []
            for j in range(len(batch)):
                pred_view = center_crop(pred_points[j].numpy()[0])
                gt_view = center_crop(gt_points[j].numpy()[0])
                valid = center_crop(masks[j].numpy()[0]).astype(bool)
                valid &= np.isfinite(pred_view).all(axis=-1)
                valid &= np.isfinite(gt_view).all(axis=-1)
                pred_parts.append(pred_view[valid])
                gt_parts.append(gt_view[valid])

            pred_array = np.concatenate(pred_parts, axis=0)
            gt_array = np.concatenate(gt_parts, axis=0)
            if len(pred_array) > args.max_points:
                pred_array = pred_array[
                    rng.choice(len(pred_array), args.max_points, replace=False)
                ]
            if len(gt_array) > args.max_points:
                gt_array = gt_array[
                    rng.choice(len(gt_array), args.max_points, replace=False)
                ]

            row = evaluate_pointcloud_pair(pred_array, gt_array, icp_threshold=0.1)
            row["scene_id"] = batch[-1]["label"][0].rsplit("/", 1)[0]
            row["inference_time_ms"] = inference_time_ms
            rows.append(row)
            log_handle.write(json.dumps(row, sort_keys=True) + "\n")
            log_handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = summarize_metrics(rows)
    summary["mean_inference_time_ms"] = float(
        np.mean([row["inference_time_ms"] for row in rows])
    )
    summary["peak_vram_mib"] = float(
        torch.cuda.max_memory_reserved() / (1024**2)
    )
    summary["kf"] = args.kf
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
