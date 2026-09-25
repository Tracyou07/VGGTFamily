"""Predict v10 windows from an immutable v9 Scene20 CPU image tensor."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from experiments.ours_v7.backend_profiles import early_prepare
_EARLY_PROFILE = early_prepare(sys.argv[1:]) if __name__ == "__main__" else None

import numpy as np
import torch

from experiments.ours_v6.runtime import preflight, sha256, source_identity, write_json
from experiments.ours_v7.diagnostic_export import save_npz_fast
from experiments.ours_v9.vkitti_131 import scene20_windows
from experiments.ours_v9.vkitti_predict import prediction_values, tensor_sha256
from vggt.v10.attention import MODES


def validate_frozen_input(root, frames):
    root=Path(root).resolve()
    source=json.loads((root/"input_manifest.json").read_text())
    if (source.get("dataset"),source.get("scene")) != ("Virtual KITTI 1.3.1","Scene20"):
        raise ValueError("expected frozen Virtual KITTI 1.3.1 Scene20 input")
    if source.get("condition") not in ("clone","rain","fog"):
        raise ValueError("unexpected Scene20 condition")
    source_input=Path(source["input_path"]).resolve()
    if source_input != (root/"inputs.pt").resolve() or sha256(source_input)!=source["input_sha256"]:
        raise ValueError("frozen input file identity changed")
    saved=torch.load(source_input,map_location="cpu",weights_only=True)
    original_ids=list(source["frame_ids"])
    if saved["frame_ids"]!=original_ids or len(saved["images"])!=len(original_ids):
        raise ValueError("frozen frame IDs or tensor length changed")
    if tensor_sha256(saved["images"])!=source["image_tensor_sha256"]:
        raise ValueError("frozen image tensor hash changed")
    if not 3<=frames<=len(original_ids):
        raise ValueError("invalid fixed frame count")
    if frames==837 and len(original_ids)!=837:
        raise ValueError("full Scene20 requires 837 frames")
    ids=original_ids[:frames]
    scene20_windows(ids)
    checkpoint=Path(source["checkpoint"])
    if sha256(checkpoint)!=source["checkpoint_sha256"]:
        raise ValueError("checkpoint identity changed")
    return saved,source,ids


def execute(args):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.v10.model import WindowReconstructor
    from experiments.ours_v7.backend_profiles import apply_backend_profile,snapshot
    from experiments.ours_v8.stream_independent import run_independent_window
    if _EARLY_PROFILE!="native_vggt":
        raise ValueError("native_vggt profile must be prepared before CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must match physical --gpu")
    target=args.output_root/args.mode
    if target.exists():
        raise FileExistsError(target)
    preflight_result=preflight(args.gpu,args.output_root,min_disk_gib=10)
    apply_backend_profile("native_vggt",torch)
    torch.cuda.init();torch.cuda.reset_peak_memory_stats()  # once for this fresh process
    torch.manual_seed(2026);np.random.seed(2026)
    read_start=time.perf_counter()
    saved,source,ids=validate_frozen_input(args.frozen_input_root,args.frames)
    images=saved["images"][:args.frames]
    windows=scene20_windows(ids)
    input_read_seconds=time.perf_counter()-read_start
    identity=source_identity()
    if identity["status"]:
        raise RuntimeError("v10 prediction requires a committed clean worktree")
    args.output_root.mkdir(parents=True,exist_ok=True)
    target.mkdir(exist_ok=False)
    model_start=time.perf_counter()
    model=VGGT().eval().requires_grad_(False)
    weights=load_file(source["checkpoint"])
    model.load_state_dict(weights,strict=True);del weights
    model.cuda();torch.cuda.synchronize()
    model_load_seconds=time.perf_counter()-model_start
    wrapper=WindowReconstructor(model)
    prediction_files=[]
    export_seconds=0.
    forward_start=time.perf_counter()
    if args.mode=="independent":
        # Preserve v8's verified one-window independent execution.
        correspondence=dict(mode="independent",pair_count=0)
        forward_seconds=0.
        for index,(lo,hi) in enumerate(windows):
            window_start=time.perf_counter()
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                prediction,_,_,_=run_independent_window(wrapper,images,ids,lo,hi,
                    60,10,512,reuse_image_encoding=False,cache_local_kv_dtype=True,
                    correspondence_attention_path="native_sdpa",
                    dense_head_frame_chunk=None)
            torch.cuda.synchronize()
            forward_seconds+=time.perf_counter()-window_start
            started=time.perf_counter()
            values=prediction_values(prediction,ids[lo:hi])
            folder=target/"windows"/f"{index:04d}"
            folder.mkdir(parents=True,exist_ok=False)
            path=folder/"local.npz"
            save_npz_fast(path,**values)
            export_seconds+=time.perf_counter()-started
            prediction_files.append(dict(window=index,lo=lo,hi=hi,
                path=str(path),sha256=sha256(path),bytes=path.stat().st_size,
                fields=list(values)))
            del prediction,values
    else:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            result=wrapper(images,ids,mode=args.mode,window_size=60,overlap=10,
                query_chunk_size=512,reuse_image_encoding=False,
                cache_local_kv_dtype=True,correspondence_attention_path="native_sdpa",
                dense_head_frame_chunk=None)
        torch.cuda.synchronize()
        if result["windows"]!=windows:
            raise ValueError("v10 returned the wrong window schedule")
        forward_seconds=time.perf_counter()-forward_start
        correspondence=result["correspondence"]
        for index,(prediction,(lo,hi)) in enumerate(zip(result["predictions"],windows)):
            started=time.perf_counter()
            values=prediction_values(prediction,ids[lo:hi])
            folder=target/"windows"/f"{index:04d}"
            folder.mkdir(parents=True,exist_ok=False)
            path=folder/"local.npz"
            save_npz_fast(path,**values)
            export_seconds+=time.perf_counter()-started
            prediction_files.append(dict(window=index,lo=lo,hi=hi,
                path=str(path),sha256=sha256(path),bytes=path.stat().st_size,
                fields=list(values)))
    torch.cuda.synchronize()
    if len(prediction_files)!=len(windows):
        raise ValueError("prediction export incomplete")
    manifest=dict(dataset="Virtual KITTI 1.3.1",scene="Scene20",
        condition=source["condition"],communication_mode=args.mode,
        frame_ids=ids,windows=windows,sources=identity,
        preprocessing=saved["preprocessing"],
        frozen_input_root=str(args.frozen_input_root.resolve()),
        input_sha256=source["input_sha256"],
        image_tensor_sha256=tensor_sha256(images),
        full_image_tensor_sha256=source["image_tensor_sha256"],
        checkpoint=source["checkpoint"],checkpoint_sha256=source["checkpoint_sha256"],
        precision="bf16",configuration=dict(input=str(Path(source["input_path"]).resolve()),
            frames=args.frames,window_size=60,overlap=10,backend_profile="native_vggt",
            correspondence_attention_path="native_sdpa",query_chunk_size=512,
            cache_local_kv_dtype=True,dense_head_frame_chunk=None,
            reuse_image_encoding=False),
        backend_profile=dict(requested="native_vggt",effective=snapshot(torch)),
        preflight=preflight_result,gpu_uuid=preflight_result["gpu_uuid"],
        prediction_files=prediction_files,correspondence=correspondence,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_scope="fresh process from CUDA initialization through local.npz export; no phase reset",
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        timing=dict(model_load_seconds=model_load_seconds,
            input_read_seconds=input_read_seconds,forward_seconds=forward_seconds,
            export_seconds=export_seconds),
        no_alignment_or_gt_evaluation=True,no_point_cloud_export=True)
    write_json(target/"run_manifest.json",manifest)
    write_json(target/"COMPLETE.json",dict(status="complete",mode=args.mode))


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--mode",choices=MODES,required=True)
    parser.add_argument("--gpu",type=int,required=True)
    parser.add_argument("--frames",type=int,default=837)
    parser.add_argument("--backend-profile",choices=("native_vggt",),required=True)
    args=parser.parse_args(argv)
    if not 3<=args.frames<=837:
        parser.error("--frames must be between 3 and 837")
    return args


def main():
    args=parse_args()
    target=args.output_root/args.mode
    if target.exists():
        raise FileExistsError(target)
    try:
        execute(args)
    except Exception as error:
        target.mkdir(parents=True,exist_ok=True)
        write_json(target/"FAILED.json",dict(reason=str(error),
            traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    main()
