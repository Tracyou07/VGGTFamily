"""Summarize frozen v9 baselines and v10 Scene20 research runs."""
import argparse
import csv
import json
from pathlib import Path

from experiments.ours_v6.runtime import sha256,write_json

MODES=("independent","camera_only","overlap_correspondence","camera_global_overlap")
BASELINE={"independent","overlap_correspondence"}
GIB=2**30


def _read(path):
    return json.loads(Path(path).read_text())


def _paths(spec,condition,mode):
    paths=spec["conditions"][condition]
    if mode in BASELINE:
        prediction=Path(paths["v9_predictions"])/mode
        alignment=Path(paths["v9_alignment"])/f"{mode}_sparse_point_camera_joint"
    else:
        prediction=Path(paths["v10_predictions"])/mode
        alignment=Path(paths["v10_alignment"])/mode
    return prediction,alignment


def collect(spec):
    rows,edges,sources=[],[],[]
    for condition in ("clone","rain","fog"):
        baseline=None
        for mode in MODES:
            prediction,alignment=_paths(spec,condition,mode)
            if not (prediction/"COMPLETE.json").is_file() or not (alignment/"COMPLETE.json").is_file():
                raise ValueError(f"incomplete {condition}/{mode}")
            pm_path=prediction/"run_manifest.json"
            am_path=alignment/"alignment_comparison.json"
            pm,am=_read(pm_path),_read(am_path)
            evaluation=_read(alignment/"evaluation_summary.json")
            if len(pm["frame_ids"])!=837 or len(pm["windows"])!=17 or len(am["edges"])!=16:
                raise ValueError(f"wrong Scene20 frame/window/edge count: {condition}/{mode}")
            if am["trajectory_metrics"]["protocol_id"]!="virtual-kitti-1.3.1-ate-sim3-v1":
                raise ValueError("GT protocol changed")
            identity={key:pm[key] for key in ("input_sha256","image_tensor_sha256",
                                              "checkpoint_sha256","frame_ids","windows","precision")}
            config=pm["configuration"]
            for key in ("window_size","overlap","backend_profile",
                        "correspondence_attention_path","query_chunk_size",
                        "cache_local_kv_dtype","dense_head_frame_chunk",
                        "reuse_image_encoding"):
                identity[key]=config[key]
            if baseline is None:baseline=identity
            elif identity!=baseline:
                raise ValueError(f"input/algorithm configuration mismatch: {condition}/{mode}")
            fwd=pm.get("forward_seconds",pm.get("timing",{}).get("forward_seconds"))
            if fwd is None:raise ValueError("forward timing missing")
            camera_bank=pm.get("correspondence",{}).get("camera_bank",{})
            summary=evaluation
            boundary=summary["ownership_boundaries"]
            if boundary["count"]!=16 or summary["adjacent"]["count"]!=836:
                raise ValueError("adjacent/boundary counts wrong")
            row=dict(condition=condition,mode=mode,
                ate_rmse_m=summary["ate_rmse_m"],
                adjacent_translation_rmse_m=summary["adjacent"]["translation_rmse_m"],
                adjacent_rotation_rmse_deg=summary["adjacent"]["rotation_rmse_deg"],
                boundary_translation_rmse_m=boundary["translation_rmse_m"],
                boundary_rotation_rmse_deg=boundary["rotation_rmse_deg"],
                forward_seconds=fwd,stitch_seconds=am["alignment_seconds"],
                reconstruction_seconds=fwd+am["alignment_seconds"],
                peak_allocated_gib=pm["peak_allocated_bytes"]/GIB,
                peak_reserved_gib=pm["peak_reserved_bytes"]/GIB,
                under_24_gib=(pm["peak_allocated_bytes"]<24*GIB and
                              pm["peak_reserved_bytes"]<24*GIB),
                fallback_edges=json.dumps(am["fallback_edges"]),
                gpu_uuid=pm["gpu_uuid"],frame_count=837,window_count=17,edge_count=16)
            row["duplicate_window_frame_camera_instances"]=camera_bank.get(
                "duplicate_window_frame_instances",0)
            row["max_remote_camera_bank_tokens"]=camera_bank.get("max_remote_tokens",0)
            rows.append(row)
            for edge in am["edges"]:
                point=edge["all_point_diagnostics"]
                edges.append(dict(condition=condition,mode=mode,
                    edge_index=edge["edge_index"],
                    selected_pairs=edge["selected_pairs"],fit_pairs=edge["fit_pairs"],
                    fallback=edge["fallback"],fallback_reason=edge["fallback_reason"],
                    scale=edge["scale"],
                    boundary_translation_error_m=edge["boundary_translation_error_m"],
                    boundary_rotation_error_deg=edge["boundary_rotation_error_deg"],
                    point_residual_rmse=point["point_residual_rmse"],
                    camera_center_residual_mean=point["center_residual_mean"],
                    camera_rotation_residual_mean_rad=point["rotation_residual_mean_rad"],
                    alignment_add_seconds=edge["add_seconds"],
                    alignment_incremental_peak_rss_bytes=edge["alignment_incremental_peak_rss_bytes"]))
            sources.append(dict(condition=condition,mode=mode,
                prediction_manifest=str(pm_path),prediction_manifest_sha256=sha256(pm_path),
                alignment_manifest=str(alignment/"source_manifest.json"),
                alignment_manifest_sha256=sha256(alignment/"source_manifest.json"),
                prediction_commit=pm["sources"]["commit"],
                gpu_uuid=pm["gpu_uuid"],input_sha256=pm["input_sha256"],
                image_tensor_sha256=pm["image_tensor_sha256"],
                checkpoint_sha256=pm["checkpoint_sha256"]))
            sources[-1]["camera_bank"]=camera_bank
    return rows,edges,sources


def _csv(path,rows):
    with Path(path).open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _report(rows,spec):
    index={(r["condition"],r["mode"]):r for r in rows}
    lines=["# v10 Scene20 research validation", "",
        "Virtual KITTI **1.3.1** Scene20, original 837 frames per condition,",
        "window 60 / overlap 10 (17 windows, 16 edges). The same frozen v9",
        "CPU image tensor and checkpoint were used for all four modes in each",
        "condition. v9 `independent` and `overlap_correspondence` predictions",
        "and sparse-alignment results were read without rerunning or rewriting them.","",
        "Every result uses BF16, native backend, native SDPA, query chunk 512,",
        "the original v8 heads, v9 sparse point-camera Sim(3), front-window",
        "ownership, and one whole-trajectory GT Sim(3). No GT enters fitting.","",
        "| Condition | Mode | ATE m | Adj T m | Adj R deg | Boundary T m | Boundary R deg | Forward s | Stitch s | Alloc GiB | Reserved GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| {condition} | {mode} | {ate_rmse_m:.3f} | {adjacent_translation_rmse_m:.3f} | {adjacent_rotation_rmse_deg:.3f} | {boundary_translation_rmse_m:.3f} | {boundary_rotation_rmse_deg:.3f} | {forward_seconds:.2f} | {stitch_seconds:.2f} | {peak_allocated_gib:.2f} | {peak_reserved_gib:.2f} |".format(**r))
    lines.extend(["","The boundary columns summarize all **16** ownership edges;",
        "`edge_results.csv` gives every edge, including fitted pair counts,",
        "Sim(3) scale, residuals, fallback and incremental RSS. Forward and",
        "stitch timings are separate. Each full prediction mode was timed once",
        "without profiler hooks; the small GPU operator profiler was run separately.","",
        f"The new camera modes retain {index[('clone','camera_global_overlap')]['duplicate_window_frame_camera_instances']} duplicate window-frame camera instances across the 17 windows; the largest other-window camera bank has {index[('clone','camera_global_overlap')]['max_remote_camera_bank_tokens']} tokens. Overlap camera copies stay independent. Per-window bank counts are in the prediction manifests.","",
        "## Result and limits",""])
    for condition in ("clone","rain","fog"):
        base=index[(condition,"overlap_correspondence")]
        candidate=index[(condition,"camera_global_overlap")]
        change=(candidate["ate_rmse_m"]/base["ate_rmse_m"]-1)*100
        time=candidate["forward_seconds"]-base["forward_seconds"]
        lines.append(f"- {condition.capitalize()}: camera+overlap ATE {candidate['ate_rmse_m']:.3f} m versus v9 overlap {base['ate_rmse_m']:.3f} m ({change:+.1f}%); forward {time:+.2f} s in these single runs.")
    lines.extend(["",
        "**Camera-global communication is not a stable accuracy gain.** It helps",
        "Clone and Fog ATE but severely degrades Rain. Boundary translation",
        "and rotation RMSE also worsen in the combined mode in all three",
        "conditions, despite Clone/Fog whole-trajectory ATE improvement.",
        "Rain's 16 edges completed without sparse fallback, so fallback is not",
        "the measured cause of its regression. Camera-only Fog falls back",
        "on edge 13 because joint optimization hit its iteration limit;",
        "the combined Fog mode does not fall back.","",
        "**The 24 GiB peak-reserved target fails.** The new modes reserve",
        "about 26.71 GiB in the complete prediction process, while peak",
        "allocated is about 23.37 GiB. This is a research validation version,",
        "not a configuration demonstrated to run within 24 GiB. The CPU",
        "stitcher's RSS is reported separately; prediction GPU peaks cannot be",
        "attributed to sparse alignment. Single-run timing cannot establish",
        "a repeatable speedup or slowdown beyond measurement variability.","",
        "The independent 100-frame gate showed that enabling camera exchange",
        "changes unaligned camera, depth and point outputs. Small-tensor",
        "FP32/FP64 dense-reference tests and a separate BF16 GPU operator check",
        "passed their preset tolerances. The GPU profiler observed cuDNN SDPA",
        "and a cuDNN-generated flash kernel, rather than inferring a kernel",
        "from backend flags. See the linked diagnostics in `source_manifest.json`.","",
        "Unknown from these runs: whether Rain's degradation is primarily",
        "caused by its raw camera changes, their layer-wise propagation, or",
        "their interaction with 16 edge transforms. Those causes require",
        "additional diagnostics; no algorithm change was made here.",""])
    return "\n".join(lines)


def execute(spec_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    spec=_read(spec_path)
    rows,edges,sources=collect(spec)
    output.mkdir(parents=True)
    _csv(output/"mode_results.csv",rows)
    _csv(output/"edge_results.csv",edges)
    write_json(output/"source_manifest.json",dict(
        source_spec=str(Path(spec_path).resolve()),source_spec_sha256=sha256(spec_path),
        sources=sources,operator_directory=spec["operator_directory"],
        operator_json_sha256=sha256(Path(spec["operator_directory"])/"gpu_operator_comparison.json"),
        fixed_100_prediction_root=spec["fixed_100_prediction_root"],
        fixed_100_alignment_root=spec["fixed_100_alignment_root"],
        fixed_100_raw_differences_sha256=sha256(Path(spec["fixed_100_prediction_root"])/"raw_prediction_differences.json"),
        baseline_reused_without_gpu_forward=True))
    (output/"REPORT.md").write_text(_report(rows,spec))
    write_json(output/"COMPLETE.json",dict(status="complete",modes=len(rows),edges=len(edges)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    execute(args.source_spec,args.output)


if __name__=="__main__":main()
