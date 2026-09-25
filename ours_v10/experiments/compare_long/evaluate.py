"""One shared, CPU-only trajectory protocol for Long and both ours_v5 modes."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from experiments.ours_v3.geometry import fit_sim3

BOUNDARIES = (("000059", "000060"), ("000089", "000090"))
COLUMNS = (
    "model", "frames", "precision", "window_config", "ate_rmse_m",
    "adjacent_translation_rmse_m", "adjacent_rotation_rmse_deg",
    "boundary_translation_rmse_m", "boundary_rotation_rmse_deg",
    "forward_seconds", "stitching_seconds", "reconstruction_total_seconds",
    "peak_allocated_bytes", "peak_reserved_bytes", "cpu_peak_rss_bytes",
)


def _rmse(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if len(values) else None


def _angle_deg(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def evaluate_poses(frame_ids, poses_c2w, scene_root, boundaries=BOUNDARIES):
    ids = [str(x).zfill(6) for x in frame_ids]
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if len(ids) != len(set(ids)) or poses.shape != (len(ids), 4, 4):
        raise ValueError("frame IDs or pose shape invalid")
    if not np.isfinite(poses).all():
        raise ValueError("nonfinite prediction; no silent frame filtering")
    gt = np.stack([np.loadtxt(Path(scene_root) / "pose" / f"{frame}.txt") for frame in ids])
    if gt.shape != poses.shape or not np.isfinite(gt).all():
        raise ValueError("invalid GT; no silent frame filtering")
    alignment = fit_sim3(poses[:, :3, 3], gt[:, :3, 3])
    aligned = alignment.apply(poses[:, :3, 3])
    ate = float(np.sqrt(np.mean(np.sum((aligned - gt[:, :3, 3]) ** 2, axis=1))))
    rows = []
    for i in range(1, len(ids)):
        pred = np.linalg.inv(poses[i - 1]) @ poses[i]
        truth = np.linalg.inv(gt[i - 1]) @ gt[i]
        translation = float(np.linalg.norm(alignment.scale * pred[:3, 3] - truth[:3, 3]))
        angle = _angle_deg(truth[:3, :3].T @ pred[:3, :3])
        rows.append(dict(before=ids[i - 1], after=ids[i], translation_error_m=translation,
                         rotation_error_deg=angle,
                         requested_boundary=(ids[i - 1], ids[i]) in boundaries))
    boundary_rows = [row for row in rows if row["requested_boundary"]]
    if len(boundary_rows) != len(boundaries):
        raise ValueError("fixed 59→60 and 89→90 boundaries missing")
    metrics = dict(
        frames=len(ids), ate_rmse_m=ate,
        adjacent_translation_rmse_m=_rmse([r["translation_error_m"] for r in rows]),
        adjacent_rotation_rmse_deg=_rmse([r["rotation_error_deg"] for r in rows]),
        boundary_translation_rmse_m=_rmse([r["translation_error_m"] for r in boundary_rows]),
        boundary_rotation_rmse_deg=_rmse([r["rotation_error_deg"] for r in boundary_rows]),
        boundary_errors=boundary_rows,
        alignment=dict(method="one whole-sequence proper Sim(3), camera centers",
                       scale=alignment.scale, rotation=alignment.rotation.tolist(),
                       translation=alignment.translation.tolist()),
    )
    return metrics, dict(frame_ids=np.asarray(ids), pred_raw=poses, gt_c2w=gt,
                         pred_aligned_centers=aligned, scale=np.asarray(alignment.scale),
                         rotation=alignment.rotation, translation=alignment.translation)


def evaluate_trajectory(path, scene_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as data:
        ids = data["frame_ids"]
        poses = data["c2w"]
    stage_start = time.perf_counter()
    metrics, arrays = evaluate_poses(ids, poses, scene_root)
    evaluation_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output / "evaluation.npz", **arrays)
    (output / "frame_ids.json").write_text(json.dumps([str(x) for x in arrays["frame_ids"]], indent=2))
    file_export_seconds = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    raw = arrays["pred_raw"][:, :3, 3]
    gt = arrays["gt_c2w"][:, :3, 3]
    axes[0].plot(raw[:, 0], raw[:, 2], label="raw prediction")
    axes[1].plot(arrays["pred_aligned_centers"][:, 0], arrays["pred_aligned_centers"][:, 2], label="whole-sequence Sim(3)")
    axes[1].plot(gt[:, 0], gt[:, 2], label="GT")
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=160)
    plt.close(fig)
    plotting_seconds = time.perf_counter() - stage_start
    return metrics, dict(gt_evaluation_seconds=evaluation_seconds, plotting_seconds=plotting_seconds, file_export_seconds=file_export_seconds)


def write_comparison(rows, output, provenance):
    output = Path(output)
    if len(rows) != 3 or {r["model"] for r in rows} != {"vggt_long", "ours_v5_independent", "ours_v5_camera_exchange"}:
        raise ValueError("comparison requires all three complete models")
    summary = dict(protocol="fixed scene0150_00 / 100 frames / BF16 / whole-sequence Sim(3)",
                   provenance=provenance, rows=rows,
                   interpretation="Long uses later-window ownership; ours_v5 uses first-window ownership. Raw local-window predictions and final stitched trajectories are distinct comparisons.")
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "comparison_table.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    lines = ["# VGGT-Long 与 ours_v5：固定 100 帧对照", "",
             "输入、checkpoint、BF16 与 GT 评测代码相同。每条完整轨迹只进行一次全段 Sim(3) 对齐。", "",
             "| Model | ATE (m) | Adj. trans. RMSE (m) | Adj. rot. RMSE (deg) | Recon. (s) |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['ate_rmse_m']:.6f} | {r['adjacent_translation_rmse_m']:.6f} | {r['adjacent_rotation_rmse_deg']:.6f} | {r['reconstruction_total_seconds']:.2f} |")
    lines += ["", "Long 将重叠帧归后窗口；ours_v5 将其归前窗口。因此原始窗口预测差异与拼接后轨迹差异必须分开解释。仅凭最终 ATE 不能断言 camera token 通信效果。",
              "", "各模式 evaluation.json 包含两个指定边界的平移与旋转误差；local.npz / Long 原始 chunk 保留拼接前预测。GT 不参与推理和拼接。",
              "", "计时定义：完整重建含预处理、前向与拼接/Long 检索，不含模型加载、GT 评测、绘图和文件导出；Long 原生预处理按实际 chunk 次数计入。"]
    (output / "comparison_report.md").write_text("\n".join(lines) + "\n")
