"""Shared evaluator snapshot, verbatim from existing eval_long_7scenes.evaluate_scene.

Only dependency loading is new. Metric code below is covered by exact regression.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import torch


def bootstrap(eval_root):
    import vggt  # Bind ours_v4 before adding the protocol's reference tree.
    root=Path(eval_root); fast=root/'reference/FastVGGT-main'
    for p in (root/'adapters',fast/'eval',fast):
        if str(p) not in sys.path: sys.path.append(str(p))
    name='vggt.utils.eval_utils'
    spec=importlib.util.spec_from_file_location(name,fast/'vggt/utils/eval_utils.py')
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module)
    from criterion import L21,Regr3D_t_ScaleShiftInv
    from data import SevenScenes
    return SevenScenes,Regr3D_t_ScaleShiftInv(L21,norm_mode=False,gt_scale=True)


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
