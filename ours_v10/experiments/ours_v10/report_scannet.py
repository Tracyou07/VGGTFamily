"""Summarize same-length v9/v10 ScanNet frozen-prediction comparisons."""
import argparse
import csv
import json
from pathlib import Path

from experiments.ours_v6.runtime import sha256, write_json
from experiments.ours_v10.predict_scannet import MODES, expected_windows


def read(path):
    return json.loads(Path(path).read_text())


def collect(run_root, frames):
    windows = expected_windows(frames)
    identities = []
    rows, edges, sources = [], [], []
    for mode in MODES:
        pred = Path(run_root) / mode
        align = Path(run_root) / "alignment" / mode
        if not (pred / "COMPLETE.json").is_file() or not (align / "COMPLETE.json").is_file():
            raise ValueError(f"incomplete ScanNet prediction/alignment: {mode}")
        prediction_manifest = pred / "run_manifest.json"
        alignment_result = align / "alignment_comparison.json"
        pm, am, metrics = read(prediction_manifest), read(alignment_result), read(align / "evaluation_summary.json")
        if (pm["dataset"], pm["scene"], pm["communication_mode"]) != ("ScanNet", "scene0000_00", mode):
            raise ValueError("dataset, scene or mode mismatch")
        if pm["frame_ids"] != [f"{i:06d}" for i in range(frames)] or pm["windows"] != [list(x) for x in windows]:
            raise ValueError("frame/window mismatch")
        if len(am["edges"]) != len(windows) - 1 or metrics["ownership_boundaries"]["count"] != len(windows) - 1:
            raise ValueError("missing alignment edge or ownership boundary")
        if metrics["adjacent"]["count"] != frames - 1 or metrics["gt_alignment"] != "one whole-trajectory proper Sim(3); never per window":
            raise ValueError("evaluation protocol mismatch")
        config = pm["configuration"]
        identity = (pm["input_sha256"], pm["image_tensor_sha256"], pm["checkpoint_sha256"],
                    pm["frame_ids"], pm["windows"], pm["precision"],
                    *(config[key] for key in ("window_size", "overlap", "backend_profile",
                        "correspondence_attention_path", "query_chunk_size",
                        "cache_local_kv_dtype", "dense_head_frame_chunk", "reuse_image_encoding")))
        identities.append(identity)
        row = dict(frames=frames, mode=mode, ate_rmse_m=metrics["ate_rmse_m"],
            adjacent_translation_rmse_m=metrics["adjacent"]["translation_rmse_m"],
            adjacent_rotation_rmse_deg=metrics["adjacent"]["rotation_rmse_deg"],
            boundary_translation_rmse_m=metrics["ownership_boundaries"]["translation_rmse_m"],
            boundary_rotation_rmse_deg=metrics["ownership_boundaries"]["rotation_rmse_deg"],
            forward_seconds=pm["timing"]["forward_seconds"],
            stitch_seconds=am["alignment_seconds"],
            peak_allocated_gib=pm["peak_allocated_bytes"] / 2**30,
            peak_reserved_gib=pm["peak_reserved_bytes"] / 2**30,
            prediction_cpu_peak_rss_gib=pm["cpu_peak_rss_bytes"] / 2**30,
            alignment_cpu_peak_rss_gib=am["process_peak_rss_bytes"] / 2**30,
            fallback_edges=json.dumps(am["fallback_edges"]),
            under_24_gib=pm["peak_allocated_bytes"] < 24*2**30 and pm["peak_reserved_bytes"] < 24*2**30)
        rows.append(row)
        for edge in am["edges"]:
            edges.append(dict(frames=frames, mode=mode, edge_index=edge["edge_index"],
                scale=edge["scale"], fallback=edge["fallback"],
                fallback_reason=edge["fallback_reason"],
                selected_pairs=edge["selected_pairs"],
                boundary_translation_error_m=edge["boundary_translation_error_m"],
                boundary_rotation_error_deg=edge["boundary_rotation_error_deg"],
                add_seconds=edge["add_seconds"]))
        sources.append(dict(mode=mode, prediction_manifest=str(prediction_manifest),
            prediction_manifest_sha256=sha256(prediction_manifest),
            alignment_result=str(alignment_result),
            alignment_result_sha256=sha256(alignment_result)))
    if identities[0] != identities[1]:
        raise ValueError("v9/v10 inputs or noncommunication configuration differ")
    baseline = rows[0]
    candidate = rows[1]
    deltas = {key: candidate[key] - baseline[key] for key in (
        "ate_rmse_m", "adjacent_translation_rmse_m", "adjacent_rotation_rmse_deg",
        "boundary_translation_rmse_m", "boundary_rotation_rmse_deg", "forward_seconds",
        "stitch_seconds", "peak_allocated_gib", "peak_reserved_gib")}
    return rows, edges, sources, deltas


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, choices=(100, 300, 500, 1000), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, edges, sources, deltas = collect(args.run_root, args.frames)
    args.output.mkdir(parents=True)
    write_csv(args.output / "comparison.csv", rows)
    write_csv(args.output / "edges.csv", edges)
    write_json(args.output / "source_manifest.json", dict(sources=sources, frames=args.frames))
    write_json(args.output / "same_length_delta.json", deltas)
    report = [f"# ScanNet scene0000_00, {args.frames} frames", "",
        "Both modes used the same frozen input prefix and checkpoint. GT was used only after sparse stitching; the trajectory received one whole-sequence Sim(3).",
        "", "| Metric | v9 overlap | v10 camera+overlap | v10 - v9 |",
        "|---|---:|---:|---:|"]
    labels = (("ATE (m)", "ate_rmse_m"), ("Adjacent translation RMSE (m)", "adjacent_translation_rmse_m"),
        ("Adjacent rotation RMSE (deg)", "adjacent_rotation_rmse_deg"),
        ("Boundary translation RMSE (m)", "boundary_translation_rmse_m"),
        ("Boundary rotation RMSE (deg)", "boundary_rotation_rmse_deg"),
        ("Forward (s)", "forward_seconds"), ("Sparse stitching (s)", "stitch_seconds"),
        ("Peak allocated (GiB)", "peak_allocated_gib"),
        ("Peak reserved (GiB)", "peak_reserved_gib"))
    for label, key in labels:
        report.append(f"| {label} | {rows[0][key]:.6f} | {rows[1][key]:.6f} | {deltas[key]:+.6f} |")
    report += ["", f"24 GiB goal: v9 {rows[0]['under_24_gib']}; v10 {rows[1]['under_24_gib']}.",
        f"Fallback edges: v9 {rows[0]['fallback_edges']}; v10 {rows[1]['fallback_edges']}.",
        "Per-edge scales, errors and timings are in edges.csv.", ""]
    (args.output / "REPORT.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
