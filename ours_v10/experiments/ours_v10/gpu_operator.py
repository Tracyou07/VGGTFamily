"""Small BF16 H20 operator check; profiler results are never formal timings."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

from experiments.ours_v7.backend_profiles import early_prepare
_EARLY_PROFILE=early_prepare(sys.argv[1:]) if __name__=="__main__" else None

import torch

from experiments.ours_v6.runtime import ROOT,preflight,sha256,source_identity,write_json
from experiments.ours_v7.backend_profiles import apply_backend_profile,snapshot
from vggt.v10.attention import build_correspondences,global_step

BF16_ATOL=3e-2
BF16_RTOL=3e-2


def execute(args):
    if args.output.exists():raise FileExistsError(args.output)
    if _EARLY_PROFILE!="native_vggt":raise ValueError("native profile must be applied early")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must match --gpu")
    device=preflight(args.gpu,args.output.parent,min_disk_gib=5)
    apply_backend_profile("native_vggt",torch)
    torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True)
    oracle=ROOT/"tests/ours_v10/test_attention.py"
    namespace=runpy.run_path(str(oracle))
    fixture,dense=namespace["fixture"],namespace["independent_dense"]
    windows=[(0,3),(2,5),(4,6)]
    ids,states_cpu,pos_cpu,tokens,block_cpu=fixture(windows,torch.float32)
    block=block_cpu.cuda().to(torch.bfloat16).eval().requires_grad_(False)
    states=[x.cuda().to(torch.bfloat16) for x in states_cpu]
    positions=[p.cuda() for p in pos_cpu]
    mapping=build_correspondences(ids,windows,[ids[lo:hi] for lo,hi in windows],
        [(1,2)]*len(windows),4,1,1,torch.device("cuda"),
        "camera_global_overlap")
    rows=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        _,_,reference,_=dense(block,states,positions,tokens,"camera_global_overlap")
        for cache in (False,True):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()  # isolated tiny operator measurement
            actual=global_step(block,states,positions,mapping,"camera_global_overlap",
                query_chunk_size=3,cache_local_kv_dtype=cache,
                attention_path="native_sdpa")
            torch.cuda.synchronize()
            allocated=torch.cuda.max_memory_allocated()
            reserved=torch.cuda.max_memory_reserved()
            differences=[(a.float()-b.float()).abs() for a,b in zip(actual,reference)]
            finite=all(bool(torch.isfinite(a).all()) for a in actual)
            accepted=all(bool(torch.allclose(a.float(),b.float(),atol=BF16_ATOL,rtol=BF16_RTOL))
                         for a,b in zip(actual,reference))
            rows.append(dict(cache_local_kv_dtype=cache,finite=finite,
                within_predeclared_tolerance=accepted,
                max_absolute_difference=max(float(d.max()) for d in differences),
                mean_absolute_difference=float(torch.cat([d.flatten() for d in differences]).mean()),
                peak_allocated_bytes=allocated,peak_reserved_bytes=reserved))
        from torch.profiler import ProfilerActivity,profile
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as profiler:
            global_step(block,states,positions,mapping,"camera_global_overlap",
                query_chunk_size=3,cache_local_kv_dtype=True,
                attention_path="native_sdpa")
            torch.cuda.synchronize()
    events=[dict(name=e.key,count=e.count,cpu_time_us=e.cpu_time_total,
                 cuda_time_us=getattr(e,"device_time_total",None))
            for e in profiler.key_averages()
            if any(key in e.key.lower() for key in
                   ("scaled_dot_product","flash","cudnn_attention","efficient_attention"))]
    result=dict(status="success",gpu=device,source=source_identity(),
        oracle_path=str(oracle),oracle_sha256=sha256(oracle),
        backend=snapshot(torch),dtype="torch.bfloat16",
        tolerance=dict(atol=BF16_ATOL,rtol=BF16_RTOL,fixed_before_gpu_measurement=True),
        windows=windows,camera_bank=[dict(remote_tokens=b["remote_tokens"],
            remote_duplicate_frame_instances=b["remote_duplicate_frame_instances"],
            remote_also_local_frame_instances=b["remote_also_local_frame_instances"])
            for b in mapping["camera_bank"]],
        comparison=rows,actual_sdpa_profiler_events=events,
        timing_scope="diagnostic profiler only; excluded from formal timing")
    write_json(args.output/"gpu_operator_comparison.json",result)
    if not events or not all(row["finite"] and row["within_predeclared_tolerance"] for row in rows):
        raise RuntimeError("GPU operator path or numerical gate failed")
    write_json(args.output/"COMPLETE.json",dict(status="complete"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--backend-profile",choices=("native_vggt",),required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    try:execute(args)
    except Exception as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/"FAILED.json",dict(reason=str(error),
            traceback=traceback.format_exc()))
        raise


if __name__=="__main__":main()
