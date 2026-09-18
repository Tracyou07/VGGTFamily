"""Evaluate the VGGT-Long chunked VGGT core on the shared 7-Scenes protocol."""

import argparse
from registered_data import DEFAULT_DATA_ROOT, prepare_evaluation
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


ADAPTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTER_DIR.parents[2]
LONG_ROOT = PROJECT_ROOT / "vggtlong"
BASE_MODELS = LONG_ROOT / "base_models"
FAST_ROOT = PROJECT_ROOT / "eval" / "7scenes" / "reference" / "FastVGGT-main"
sys.path[:0] = [str(BASE_MODELS)]

from vggt.models.vggt import VGGT as BaseVGGT
from vggt.utils.load_fn import load_and_preprocess_images

sys.path.insert(0, str(FAST_ROOT / "eval"))
sys.path.insert(0, str(FAST_ROOT))
eval_utils_spec = importlib.util.spec_from_file_location(
    "vggt.utils.eval_utils", FAST_ROOT / "vggt" / "utils" / "eval_utils.py"
)
eval_utils_module = importlib.util.module_from_spec(eval_utils_spec)
sys.modules["vggt.utils.eval_utils"] = eval_utils_module
eval_utils_spec.loader.exec_module(eval_utils_module)
from criterion import L21, Regr3D_t_ScaleShiftInv
from data import SevenScenes


class ProtocolVGGT(BaseVGGT):
    def to(self, *args, **kwargs):
        if args == (torch.bfloat16,) and not kwargs:
            return self
        if not args and kwargs.get("dtype") is torch.bfloat16 and "device" not in kwargs:
            return self
        return super().to(*args, **kwargs)


def estimate_sim3(src, dst, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    weights /= weights.sum()
    src_mean = (src * weights[:, None]).sum(axis=0)
    dst_mean = (dst * weights[:, None]).sum(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c * weights[:, None]).T @ src_c
    u, _, vt = np.linalg.svd(cov)
    sgn = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sgn[-1, -1] = -1
    rot = u @ sgn @ vt
    var = (weights * (src_c * src_c).sum(axis=1)).sum()
    scale = float(np.trace(np.diag(np.linalg.svd(cov, compute_uv=False)) @ sgn) / max(var, 1e-12))
    trans = dst_mean - scale * (rot @ src_mean)
    return scale, rot, trans


def align_chunk(reference, reference_conf, current_overlap, current_overlap_conf, current, current_conf):
    threshold = min(float(np.median(reference_conf)), float(np.median(current_overlap_conf))) * 0.1
    valid = (
        (reference_conf > threshold)
        & (current_overlap_conf > threshold)
        & np.isfinite(reference).all(axis=-1)
        & np.isfinite(current_overlap).all(axis=-1)
    )
    src = current_overlap[valid]
    dst = reference[valid]
    weights = np.sqrt(reference_conf[valid] * current_overlap_conf[valid])
    if len(src) > 100000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(src), 100000, replace=False)
        src, dst, weights = src[idx], dst[idx], weights[idx]
    if len(src) < 3:
        raise RuntimeError(f"insufficient overlap correspondences: {len(src)}")
    scale, rot, trans = estimate_sim3(src, dst, weights)
    return scale * np.einsum("ij,nhwj->nhwi", rot, current) + trans


def evaluate_scene(gt_batch, pred_points, pred_conf, criterion):
    pred_views = [
        {
            "pts3d_in_other_view": torch.from_numpy(pred_points[i])[None],
            "conf": torch.from_numpy(pred_conf[i])[None],
        }
        for i in range(len(pred_points))
    ]
    gt_pts, pred_pts, _, _, masks, _ = criterion.get_all_pts3d_t(gt_batch, pred_views)
    pred_parts, gt_parts = [], []
    for i in range(len(gt_batch)):
        image_h, image_w = gt_batch[i]["img"].shape[-2:]
        cx, cy = image_w // 2, image_h // 2
        left, top = cx - 112, cy - 112
        right, bottom = cx + 112, cy + 112
        pred = pred_pts[i].numpy()[0][top:bottom, left:right]
        gt = gt_pts[i].numpy()[0][top:bottom, left:right]
        valid = masks[i].numpy()[0][top:bottom, left:right].astype(bool)
        valid &= np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
        pred_parts.append(pred[valid])
        gt_parts.append(gt[valid])
    pred = np.concatenate(pred_parts, axis=0)
    gt = np.concatenate(gt_parts, axis=0)
    if len(pred) > 999999:
        pred = pred[np.random.default_rng(0).choice(len(pred), 999999, replace=False)]
    if len(gt) > 999999:
        gt = gt[np.random.default_rng(1).choice(len(gt), 999999, replace=False)]

    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
    reg = o3d.pipelines.registration.registration_icp(
        pred_cloud, gt_cloud, 0.1, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_cloud.transform(reg.transformation)
    pred_cloud.estimate_normals()
    gt_cloud.estimate_normals()
    pred = np.asarray(pred_cloud.points)
    gt = np.asarray(gt_cloud.points)
    pred_normals = np.asarray(pred_cloud.normals)
    gt_normals = np.asarray(gt_cloud.normals)
    gt_tree = o3d.geometry.KDTreeFlann(gt_cloud)
    pred_tree = o3d.geometry.KDTreeFlann(pred_cloud)
    acc_d, acc_n, acc_nc = [], [], []
    for p, n in zip(pred, pred_normals):
        _, idx, dist = gt_tree.search_knn_vector_3d(p, 1)
        if idx:
            acc_d.append(np.sqrt(dist[0]))
            acc_nc.append(abs(float(np.dot(n, gt_normals[idx[0]]))))
    comp_d, comp_nc = [], []
    for p, n in zip(gt, gt_normals):
        _, idx, dist = pred_tree.search_knn_vector_3d(p, 1)
        if idx:
            comp_d.append(np.sqrt(dist[0]))
            comp_nc.append(abs(float(np.dot(n, pred_normals[idx[0]]))))
    return {
        "acc": float(np.mean(acc_d)),
        "comp": float(np.mean(comp_d)),
        "nc1": float(np.mean(acc_nc)),
        "nc2": float(np.mean(comp_nc)),
    }


def run(args):
    prepare_evaluation(args.data_root, args.output_dir, kf=args.kf)
    dataset = SevenScenes(
        split="test", ROOT=args.data_root, resolution=(518, 392),
        num_seq=1, full_video=True, kf_every=args.kf,
    )
    checkpoint = Path(args.checkpoint)
    from safetensors.torch import load_file
    model = ProtocolVGGT().cuda().eval()
    state = load_file(str(checkpoint), device="cpu") if checkpoint.suffix == ".safetensors" else torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    del state
    model = model.cuda().eval()
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for data_idx in range(len(dataset)):
        batch = __import__("torch.utils.data._utils.collate", fromlist=["default_collate"]).default_collate([dataset[data_idx]])
        image_paths = [view["instance"][0] for view in batch]
        chunks = []
        start = 0
        while True:
            end = min(start + args.chunk_size, len(image_paths))
            chunks.append((start, end))
            if end == len(image_paths):
                break
            start = end - args.overlap
        aligned_parts, conf_parts = [], []
        previous = None
        for chunk_idx, (start, end) in enumerate(chunks):
            images = load_and_preprocess_images(image_paths[start:end]).cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                prediction = model(images)
            torch.cuda.synchronize()
            if chunk_idx == 0 and data_idx == 0:
                print(f"VGGT-Long model={type(model).__module__}.{type(model).__name__}", flush=True)
                print(f"VGGT-Long prediction keys={sorted(prediction.keys())}", flush=True)
            if chunk_idx == 0:
                inference_time_ms = (time.perf_counter() - t0) * 1000.0
            points = prediction["world_points"].float().cpu().numpy()[0]
            conf = prediction["world_points_conf"].float().cpu().numpy()[0]
            if previous is not None:
                points = align_chunk(
                    previous[0][-args.overlap:], previous[1][-args.overlap:],
                    points[:args.overlap], conf[:args.overlap], points, conf,
                )
            previous = (points, conf)
            keep = slice(None) if chunk_idx == 0 else slice(args.overlap, None)
            aligned_parts.append(points[keep])
            conf_parts.append(conf[keep])
        pred_points = np.concatenate(aligned_parts, axis=0)
        pred_conf = np.concatenate(conf_parts, axis=0)
        row = evaluate_scene(batch, pred_points, pred_conf, criterion)
        row["scene_id"] = batch[-1]["label"][0].rsplit("/", 1)[0]
        row["inference_time_ms"] = inference_time_ms
        rows.append(row)
        with (out / "sequences.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("acc", "comp", "nc1", "nc2", "inference_time_ms")}
    summary["mean_nc"] = (summary["nc1"] + summary["nc2"]) / 2
    summary["valid_sequences"] = len(rows)
    summary["peak_vram_mib"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
    summary["kf"] = args.kf
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kf", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=30)
    run(parser.parse_args())
