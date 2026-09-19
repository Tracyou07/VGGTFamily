import numpy as np
from scipy.spatial import cKDTree


def deterministic_sample(points, cap, seed):
    points = np.asarray(points)
    if len(points) <= cap:
        return points.copy()
    return points[np.random.default_rng(seed).choice(len(points), cap, replace=False)]


def lower_nanmedian(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan
    return np.partition(finite, (len(finite) - 1) // 2)[(len(finite) - 1) // 2]


def _lower_median_rows(points):
    points = np.asarray(points)
    k = (len(points) - 1) // 2
    return np.partition(points, k, axis=0)[k]


def normalize_views_like_fastvggt(pred_views, gt_views, masks):
    pred = [np.asarray(x, dtype=np.float64).copy() for x in pred_views]
    gt = [np.asarray(x, dtype=np.float64).copy() for x in gt_views]
    masks = [np.asarray(x, dtype=bool) for x in masks]
    pred_valid = np.concatenate([x[m] for x, m in zip(pred, masks)])
    gt_valid = np.concatenate([x[m] for x, m in zip(gt, masks)])
    if not len(pred_valid) or not len(gt_valid):
        raise ValueError("non-empty correspondences required")
    pred_shift = lower_nanmedian(pred_valid[:, 2])
    gt_shift = lower_nanmedian(gt_valid[:, 2])
    for x in pred:
        x[..., 2] -= pred_shift
    for x in gt:
        x[..., 2] -= gt_shift
    pred_valid = np.concatenate([x[m] for x, m in zip(pred, masks)])
    gt_valid = np.concatenate([x[m] for x, m in zip(gt, masks)])
    pred_center = _lower_median_rows(pred_valid)
    gt_center = _lower_median_rows(gt_valid)
    pred_scale = lower_nanmedian(np.linalg.norm(pred_valid - pred_center, axis=1))
    gt_scale = lower_nanmedian(np.linalg.norm(gt_valid - gt_center, axis=1))
    if pred_scale < 1e-8 or gt_scale < 1e-8:
        raise ValueError("degenerate cloud scale")
    factor = gt_scale / float(np.clip(pred_scale, 1e-3, 1e3))
    pred = [x * factor for x in pred]
    return pred, gt


def fastvggt_scale_shift_clouds(pred, gt):
    p = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    g = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    mask_p = np.isfinite(p).all(1)
    mask_g = np.isfinite(g).all(1)
    if mask_p.sum() == 0 or mask_g.sum() == 0:
        raise ValueError("non-empty finite clouds required")
    # Aggregate SLAM outputs have no pixel correspondence. This is the exact
    # FastVGGT shift/scale statistic applied independently to the two clouds.
    p = p[mask_p].copy()
    g = g[mask_g].copy()
    p[:, 2] -= lower_nanmedian(p[:, 2])
    g[:, 2] -= lower_nanmedian(g[:, 2])
    pc = _lower_median_rows(p)
    gc = _lower_median_rows(g)
    ps = lower_nanmedian(np.linalg.norm(p - pc, axis=1))
    gs = lower_nanmedian(np.linalg.norm(g - gc, axis=1))
    if ps < 1e-8 or gs < 1e-8:
        raise ValueError("degenerate cloud scale")
    return p * (gs / float(np.clip(ps, 1e-3, 1e3))), g


def score_clouds(pred, gt, seed=0, point_cap=999999, icp_threshold=0.1):
    import open3d as o3d

    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    pred = pred[np.isfinite(pred).all(1)]
    gt = gt[np.isfinite(gt).all(1)]
    if not len(pred) or not len(gt):
        raise ValueError("point clouds must be non-empty and finite")
    pred = deterministic_sample(pred, point_cap, seed)
    gt = deterministic_sample(gt, point_cap, seed + 1)
    pred_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
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
    pred = np.asarray(pred_cloud.points)
    gt = np.asarray(gt_cloud.points)
    pred_normals = np.asarray(pred_cloud.normals)
    gt_normals = np.asarray(gt_cloud.normals)
    acc_dist, acc_idx = cKDTree(gt).query(pred, workers=-1)
    comp_dist, comp_idx = cKDTree(pred).query(gt, workers=-1)
    nc1 = np.abs(np.sum(gt_normals[acc_idx] * pred_normals, axis=-1))
    nc2 = np.abs(np.sum(pred_normals[comp_idx] * gt_normals, axis=-1))
    return {
        "acc": float(np.mean(acc_dist)),
        "acc_med": float(np.median(acc_dist)),
        "comp": float(np.mean(comp_dist)),
        "comp_med": float(np.median(comp_dist)),
        "nc1": float(np.mean(nc1)),
        "nc1_med": float(np.median(nc1)),
        "nc2": float(np.mean(nc2)),
        "nc2_med": float(np.median(nc2)),
        "nc": float((np.mean(nc1) + np.mean(nc2)) / 2),
        "nc_med": float((np.median(nc1) + np.median(nc2)) / 2),
    }
