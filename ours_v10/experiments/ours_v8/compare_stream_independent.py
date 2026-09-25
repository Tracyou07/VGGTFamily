"""Gate a streamed independent run against frozen unstreamed predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.ours_v6.runtime import sha256, write_json


RAW_ATOL = 2e-5
RAW_RTOL = 2e-5
GEOMETRY_ATOL = 1e-6
FIELDS = ("pose_encoding", "c2w", "intrinsics", "depth", "depth_conf",
          "confidence", "world_points", "world_points_conf")


def compare_arrays(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return dict(shape_match=False, dtype_match=False, finite=False,
                    exact=False, within_tolerance=False, max_abs=None,
                    mean_abs=None, relative_l2=None)
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    exact = True
    within = finite
    max_abs = absolute_sum = squared_sum = reference_squared = count = 0
    for start in range(0, len(reference), 4):
        first = reference[start:start + 4]
        second = candidate[start:start + 4]
        exact &= bool(np.array_equal(first, second))
        if not finite:
            continue
        difference = second.astype(np.float64) - first.astype(np.float64)
        absolute = np.abs(difference)
        max_abs = max(max_abs, float(absolute.max()))
        absolute_sum += float(absolute.sum())
        squared_sum += float(np.square(difference).sum())
        reference_squared += float(np.square(first.astype(np.float64)).sum())
        count += difference.size
        within &= bool(np.all(absolute <= RAW_ATOL + RAW_RTOL * np.abs(first)))
    return dict(shape_match=True, dtype_match=True, finite=finite,
                exact=exact, within_tolerance=within,
                max_abs=max_abs if finite else None,
                mean_abs=absolute_sum/count if finite else None,
                relative_l2=(squared_sum/max(reference_squared, 1e-300))**.5 if finite else None)


def compare_transform(reference, candidate):
    records = {}
    for key in ("scale", "rotation", "translation"):
        left = np.asarray(reference[key], dtype=np.float64)
        right = np.asarray(candidate[key], dtype=np.float64)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            records[key] = dict(valid=False, max_abs=None)
        else:
            difference = float(np.max(np.abs(left-right)))
            records[key] = dict(valid=difference <= GEOMETRY_ATOL, max_abs=difference)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    left_manifest = json.loads((args.reference / "run_manifest.json").read_text())
    right_manifest = json.loads((args.candidate / "run_manifest.json").read_text())
    identity = {}
    for key in ("input_sha256", "checkpoint_sha256", "attention_sha256",
                "v8_attention_sha256", "joint_alignment_sha256", "precision",
                "frame_ids", "windows"):
        identity[key] = left_manifest[key] == right_manifest[key]
    for key in ("backend_profile", "mode", "frames", "window_size", "overlap",
                "query_chunk_size", "reuse_image_encoding", "cache_local_kv_dtype",
                "correspondence_attention_path", "dense_head_frame_chunk",
                "alignment_mode", "npz_compression_level"):
        identity[f"configuration.{key}"] = (left_manifest["configuration"].get(key) ==
                                            right_manifest["configuration"].get(key))
    identity["effective_backend"] = (left_manifest["backend_profile"]["effective"] ==
                                     right_manifest["backend_profile"]["effective"])
    identity["prepared_input_file_hash"] = (
        sha256(left_manifest["configuration"]["input"]) == left_manifest["input_sha256"])

    windows = left_manifest["windows"]
    raw_rows = []
    raw_pass = True
    for window_id, (lo, hi) in enumerate(windows):
        reference_path = args.reference / "windows" / f"{window_id:04d}" / "local.npz"
        candidate_path = args.candidate / "windows" / f"{window_id:04d}" / "local.npz"
        reference_hash = sha256(reference_path)
        candidate_hash = sha256(candidate_path)
        with np.load(reference_path, allow_pickle=False) as old, np.load(candidate_path, allow_pickle=False) as new:
            if set(old.files) != set(new.files) or set(old.files) != set(FIELDS) | {"frame_ids"}:
                raise ValueError(f"window {window_id}: prediction fields changed")
            if not np.array_equal(old["frame_ids"], new["frame_ids"]):
                raise ValueError(f"window {window_id}: frame IDs differ")
            if list(new["frame_ids"]) != right_manifest["frame_ids"][lo:hi]:
                raise ValueError(f"window {window_id}: frame IDs disagree with manifest")
            for field in FIELDS:
                comparison = compare_arrays(old[field], new[field])
                raw_pass &= comparison["within_tolerance"]
                raw_rows.append(dict(window_id=window_id, first_frame=lo, end_frame=hi,
                                     field=field, reference_sha256=reference_hash,
                                     candidate_sha256=candidate_hash, **comparison))
    with (args.output / "raw_prediction_differences.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader(); writer.writerows(raw_rows)

    trajectory = {}
    with np.load(args.reference / "global_trajectory.npz", allow_pickle=False) as old, \
            np.load(args.candidate / "global_trajectory.npz", allow_pickle=False) as new:
        for key in ("frame_ids", "source_window"):
            trajectory[key] = dict(exact=bool(np.array_equal(old[key], new[key])))
        for key in ("c2w", "intrinsics"):
            trajectory[key] = compare_arrays(old[key], new[key])

    edge_rows = []
    for window_id in range(1, len(windows)):
        filename = f"edge_{window_id-1:04d}_{window_id:04d}.json"
        old = json.loads((args.reference / "alignment" / filename).read_text())
        new = json.loads((args.candidate / "alignment" / filename).read_text())
        if old["status"] != "success" or new["status"] != "success":
            raise ValueError(f"edge {window_id}: failed alignment")
        for transform_name in ("adjacent", "global_transform"):
            comparisons = compare_transform(old[transform_name], new[transform_name])
            for component, values in comparisons.items():
                edge_rows.append(dict(edge=window_id-1, transform=transform_name,
                                      component=component, **values))
    with (args.output / "edge_transform_differences.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(edge_rows[0]))
        writer.writeheader(); writer.writerows(edge_rows)

    old_metrics = left_manifest["evaluation_summary"]
    new_metrics = right_manifest["evaluation_summary"]
    metric_rows = []
    for group in ("adjacent", "within_window", "ownership_boundaries"):
        for key in ("translation_rmse_m", "rotation_rmse_deg"):
            old = old_metrics[group][key]; new = new_metrics[group][key]
            metric_rows.append(dict(group=group, metric=key, reference=old,
                                    candidate=new, abs_diff=abs(new-old),
                                    within_tolerance=abs(new-old) <= GEOMETRY_ATOL))
    old_ate = old_metrics["ate_rmse_m"]; new_ate = new_metrics["ate_rmse_m"]
    metric_rows.append(dict(group="trajectory", metric="ate_rmse_m", reference=old_ate,
                            candidate=new_ate, abs_diff=abs(new_ate-old_ate),
                            within_tolerance=abs(new_ate-old_ate) <= GEOMETRY_ATOL))
    old_adjacent = json.loads((args.reference / "adjacent_pose_errors.json").read_text())
    new_adjacent = json.loads((args.candidate / "adjacent_pose_errors.json").read_text())
    if len(old_adjacent) != len(new_adjacent):
        raise ValueError("adjacent trajectory lengths differ")
    boundary_differences = []
    for index, (old, new) in enumerate(zip(old_adjacent, new_adjacent)):
        if (old["before"], old["after"], old["boundary"]) != \
                (new["before"], new["after"], new["boundary"]):
            raise ValueError(f"adjacent frame mapping changed at {index}")
        for key in ("translation_error", "rotation_error_deg"):
            difference = abs(old[key] - new[key])
            if old["boundary"]:
                boundary_differences.append(dict(index=index, before=old["before"],
                    after=old["after"], field=key, reference=old[key], candidate=new[key],
                    abs_diff=difference, within_tolerance=difference <= GEOMETRY_ATOL))
            metric_rows.append(dict(group="adjacent_frame", metric=f"{index}.{key}",
                reference=old[key], candidate=new[key], abs_diff=difference,
                within_tolerance=difference <= GEOMETRY_ATOL))
    with (args.output / "boundary_differences.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(boundary_differences[0]))
        writer.writeheader(); writer.writerows(boundary_differences)

    expected_boundary_count = len(windows) - 1
    geometry_pass = (all(value["exact"] for key, value in trajectory.items()
                         if key in ("frame_ids", "source_window")) and
                     all(trajectory[key]["within_tolerance"] for key in ("c2w", "intrinsics")) and
                     all(row["valid"] for row in edge_rows) and
                     all(row["within_tolerance"] for row in metric_rows) and
                     len(boundary_differences) == 2 * expected_boundary_count)
    summary = dict(status="pass" if all(identity.values()) and raw_pass and geometry_pass else "fail",
        raw_atol=RAW_ATOL, raw_rtol=RAW_RTOL, geometry_atol=GEOMETRY_ATOL,
        reference=str(args.reference), candidate=str(args.candidate),
        identity=identity, raw_rows=len(raw_rows), raw_pass=raw_pass,
        raw_all_exact=all(row["exact"] for row in raw_rows),
        raw_max_abs=max(row["max_abs"] for row in raw_rows if row["max_abs"] is not None),
        trajectory=trajectory, edge_rows=len(edge_rows),
        boundary_count=len(boundary_differences)//2,
        boundary_pass=all(row["within_tolerance"] for row in boundary_differences),
        geometry_pass=geometry_pass,
        metric_differences=metric_rows[:7],
        reference_timing=left_manifest["timing"], candidate_timing=right_manifest["timing"],
        reference_memory=dict(allocated=left_manifest["peak_allocated_bytes"],
                              reserved=left_manifest["peak_reserved_bytes"]),
        candidate_memory=dict(allocated=right_manifest["peak_allocated_bytes"],
                              reserved=right_manifest["peak_reserved_bytes"]))
    write_json(args.output / "comparison_summary.json", summary)
    print(json.dumps(dict(status=summary["status"], raw_all_exact=summary["raw_all_exact"],
                          raw_max_abs=summary["raw_max_abs"],
                          boundary_count=summary["boundary_count"],
                          geometry_pass=geometry_pass), indent=2), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
