from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .data import SCENES, load_scene, preflight_dataset
from .geometry import backproject, transform_points
from .scoring import (
    score_clouds,
    fastvggt_scale_shift_clouds,
    normalize_views_like_fastvggt,
)
from .results import commit_scene, require_resume_compatible, summarize


def _fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _center_crop_mask(shape, size=224):
    mask = np.zeros(shape, dtype=bool)
    top = (shape[0] - size) // 2
    left = (shape[1] - size) // 2
    mask[top : top + size, left : left + size] = True
    return mask


def run(config, backend, resume=False, only=None):
    root = Path(config["dataset_root"])
    report = preflight_dataset(root)
    out = Path(config["output_root"]) / backend.name
    selected = [s for s in SCENES if not only or s in only]
    provenance = {
        "protocol": report["protocol"],
        "input": report["input_fingerprint"],
        "config": _fingerprint(config),
        "backend": backend.provenance(),
    }
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"nonempty output requires --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for sid in selected:
        if resume and require_resume_compatible(out, sid, provenance):
            continue
        scene = load_scene(root, sid)
        started = time.perf_counter()
        pred = backend.predict(scene.model, out / ".work" / sid)
        if pred.scene_id != sid or tuple(pred.frame_ids) != scene.model.frame_ids:
            raise ValueError("prediction frame identity/order mismatch")

        first_from_world = np.linalg.inv(scene.poses_c2w[0])
        gt_views = []
        depths = []
        for i in range(len(pred.frame_ids)):
            depth = scene.depth_m(i)
            world = backproject(depth, scene.intrinsics[i], scene.poses_c2w[i])
            gt_views.append(transform_points(world, first_from_world))
            depths.append(depth)

        if pred.aggregate_points is not None:
            full_gt = np.concatenate(
                [g[d > 0] for g, d in zip(gt_views, depths)], axis=0
            )
            pred_cloud, normalized_gt = fastvggt_scale_shift_clouds(
                pred.aggregate_points, full_gt
            )
            # Aggregate output cannot identify pixels; kept only as a compatibility
            # path for third-party adapters and is not used by bundled backends.
            gt_cloud = normalized_gt
        else:
            pred_views = []
            masks = []
            for i, (depth, gt) in enumerate(zip(depths, gt_views)):
                points = np.asarray(pred.world_points[i])
                if points.shape[:2] != depth.shape:
                    raise ValueError(
                        f"prediction shape {points.shape[:2]} != protocol shape {depth.shape}"
                    )
                mask = (
                    (depth > 0)
                    & np.asarray(pred.valid_masks[i], bool)
                    & np.isfinite(points).all(-1)
                    & np.isfinite(gt).all(-1)
                )
                pred_views.append(points)
                masks.append(mask)
            # FastVGGT normalizes on every valid full-frame pixel before the
            # 224x224 evaluation crop.
            norm_pred, norm_gt = normalize_views_like_fastvggt(
                pred_views, gt_views, masks
            )
            pred_parts = []
            gt_parts = []
            for p, g, mask in zip(norm_pred, norm_gt, masks):
                keep = mask & _center_crop_mask(mask.shape)
                pred_parts.append(p[keep])
                gt_parts.append(g[keep])
            pred_cloud = np.concatenate(pred_parts)
            gt_cloud = np.concatenate(gt_parts)

        metrics = score_clouds(
            pred_cloud,
            gt_cloud,
            seed=int(config.get("seed", 42)),
            point_cap=int(config.get("point_cap", 999999)),
            icp_threshold=float(config.get("icp_threshold", 0.1)),
        )
        metrics.update(
            inference_seconds=float(pred.inference_seconds),
            adapter_seconds=float(pred.adapter_seconds),
            scene_wall_seconds=time.perf_counter() - started,
            peak_allocated_bytes=int(pred.peak_allocated_bytes),
            peak_reserved_bytes=int(pred.peak_reserved_bytes),
        )
        commit_scene(out, sid, metrics, provenance, pred.metadata)
    # A single-scene debug run is always visibly partial against the canonical
    # nine-scene protocol.
    return summarize(out, SCENES)
