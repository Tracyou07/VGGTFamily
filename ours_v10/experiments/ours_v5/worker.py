"""One BF16 CUDA reconstruction process for one communication mode."""
import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch

from experiments.ours_v5.runtime import ROOT, fresh_directory, sha256, write_json, source_identity, preflight, write_point_cloud
from experiments.ours_v5.windows import make_windows


def execute(args):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    import vggt.layers.attention as attention_module
    from vggt.v5.model import WindowReconstructor

    config=json.loads((ROOT/'configs/v5_validation.json').read_text())
    before=preflight(args.gpu,args.output.parent)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):
        raise ValueError('CUDA_VISIBLE_DEVICES must match --gpu')
    source=source_identity()
    if not source.get('commit') or source.get('status'):
        raise RuntimeError('ours_v5 needs a committed, clean working tree')
    torch.manual_seed(2026);np.random.seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    saved=torch.load(args.input,map_location='cpu',weights_only=True)
    images=saved['images'][:args.frames];ids=saved['frame_ids'][:args.frames]
    if len(ids)!=args.frames or len(set(ids))!=len(ids):raise ValueError('invalid prepared frames')
    windows=make_windows(len(ids),args.window_size,args.overlap)
    checkpoint=Path(config['checkpoint'])
    if sha256(checkpoint)!=config['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
    manifest=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  sources=source,checkpoint=str(checkpoint),checkpoint_sha256=config['checkpoint_sha256'],
                  attention_source=attention_module.__file__,attention_sha256=sha256(attention_module.__file__),
                  preflight=before,frame_ids=ids,windows=windows,preprocessing=saved['preprocessing'],input_sha256=sha256(args.input),
                  torch=torch.__version__,cuda=torch.version.cuda,gpu_uuid=before['gpu_uuid'],precision='bf16',
                  backend='PyTorch SDPA dispatcher on allowed-edge submatrices',
                  attention_backend_flags=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                                               memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                               math=torch.backends.cuda.math_sdp_enabled()),
                  cpu_offload=False,position_convention='OpenCV w2c extrinsics, stored c2w; intrinsics in resized-image pixels',
                  point_map='original point-head world_points for Long alignment; depth-unprojection saved separately',seed=2026)
    write_json(args.output/'run_manifest.json',manifest)
    model=VGGT().eval().requires_grad_(False)
    weights=load_file(str(checkpoint));model.load_state_dict(weights,strict=True);del weights
    model.cuda();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        result=WindowReconstructor(model)(images,ids,mode=args.mode,window_size=args.window_size,overlap=args.overlap,
                                          batch_size=args.batch_size,camera_query_chunk_size=config['camera_query_chunk_size'])
    torch.cuda.synchronize();forward=time.perf_counter()-start
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    del model;torch.cuda.empty_cache()
    export_start=time.perf_counter()
    predictions=[]
    for i,pred in enumerate(result['predictions']):
        values={k:v.numpy() if torch.is_tensor(v) else v for k,v in pred.items()}
        predictions.append(values)
        folder=args.output/'windows'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'local.npz',**values)
    export_seconds=time.perf_counter()-export_start
    from vggt.v5.alignment import Stitcher,LongAlignmentConfig
    from experiments.ours_v4.artifacts import evaluate,write_cloud
    stitch=Stitcher(args.output/'alignment',LongAlignmentConfig())
    cloud=[];cloud_windows=[];stitch_seconds=0.;diagnostic_seconds=0.
    for i,(pred,(lo,hi)) in enumerate(zip(predictions,windows)):
        start=time.perf_counter();fresh,global_pred=stitch.add(pred,i);stitch_seconds+=time.perf_counter()-start
        start=time.perf_counter();folder=args.output/'windows'/f'{i:04d}'
        xyz,_=write_point_cloud(folder/'point_head_global.ply',global_pred,images[lo:hi],fresh,i)
        write_cloud(folder/'depth_unprojection_global.ply',global_pred,images[lo:hi],fresh,16,i)
        cloud.append(xyz);cloud_windows.extend([i]*len(xyz));diagnostic_seconds+=time.perf_counter()-start
    global_result=stitch.finish(ids);np.savez_compressed(args.output/'global_trajectory.npz',**global_result)
    start=time.perf_counter();evaluate(global_result,saved['scene_root'],args.output);evaluation_seconds=time.perf_counter()-start
    start=time.perf_counter()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xyz=np.concatenate(cloud);colors=np.asarray(cloud_windows)
    fig=plt.figure();ax=fig.add_subplot(projection='3d')
    ax.scatter(*xyz.T,c=colors,s=.2)
    ax.set_title('Point-head cloud; colors = source window; no GT transform')
    fig.savefig(args.output/'point_head_preview.png',dpi=150);plt.close(fig)
    diagnostic_seconds+=time.perf_counter()-start
    prep=saved['preprocessing']['elapsed_seconds']
    manifest.update(timing=dict(result['timing'],image_preprocessing_seconds=prep,forward_seconds=forward,
                                overlap_stitching_seconds=stitch_seconds,reconstruction_total_seconds=prep+forward+stitch_seconds,
                                protocol_evaluation_seconds=evaluation_seconds,export_seconds=export_seconds,
                                diagnostic_seconds=diagnostic_seconds),
                    groups=result['groups'],retained_head_features_bytes=result['retained_head_features_bytes'],
                    peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                    cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,trainable_parameters=0)
    write_json(args.output/'run_manifest.json',manifest)
    write_json(args.output/'COMPLETE.json',dict(status='complete',kind='diagnostic',mode=args.mode))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--frames',type=int,required=True)
    parser.add_argument('--window-size',type=int,default=60)
    parser.add_argument('--overlap',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--mode',choices=['independent','camera_exchange'],required=True)
    args=parser.parse_args();fresh_directory(args.output)
    try:execute(args)
    except Exception as error:
        write_json(args.output/'FAILED.json',dict(reason=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
