"""One BF16 CUDA reconstruction process for a v8 correspondence mode."""
import argparse
import json
import os
import sys
from experiments.ours_v7.backend_profiles import early_prepare
_EARLY_PROFILE = early_prepare(sys.argv[1:]) if __name__ == "__main__" else None
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch

from experiments.ours_v6.runtime import ROOT, fresh_directory, sha256, write_json, source_identity, preflight, write_point_cloud
from experiments.ours_v6.windows import make_windows


def execute(args):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    import vggt.layers.attention as attention_module
    from vggt.v8.model import WindowReconstructor
    import vggt.v8.attention as v8_attention_module
    import vggt.v8.joint_alignment as joint_alignment_module

    from experiments.ours_v7.backend_profiles import apply_backend_profile, snapshot
    profile=getattr(args,"backend_profile","legacy")
    if _EARLY_PROFILE is not None and profile!=_EARLY_PROFILE:
        raise ValueError("early backend profile differs from parsed profile")
    task_start=time.perf_counter()
    config=json.loads((ROOT/'configs/v7_validation.json').read_text())
    before=preflight(args.gpu,args.output.parent)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):
        raise ValueError('CUDA_VISIBLE_DEVICES must match --gpu')
    # Device preflight is read-only. Set numerical flags before any CUDA init.
    apply_backend_profile(profile,torch)
    whole_task=getattr(args,"whole_task_measurement",False)
    if whole_task:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()  # Only reset in this entire fresh process.
    source=source_identity()
    if not source.get('commit') or source.get('status'):
        raise RuntimeError('ours_v8 needs a committed, clean working tree')
    torch.manual_seed(2026);np.random.seed(2026)
    input_start=time.perf_counter()
    saved=torch.load(args.input,map_location='cpu',weights_only=True)
    images=saved['images'][:args.frames];ids=saved['frame_ids'][:args.frames]
    if len(ids)!=args.frames or len(set(ids))!=len(ids):raise ValueError('invalid prepared frames')
    windows=make_windows(len(ids),args.window_size,args.overlap)
    input_seconds=time.perf_counter()-input_start
    checkpoint=Path(config['checkpoint'])
    if sha256(checkpoint)!=config['checkpoint_sha256']:raise ValueError('checkpoint identity mismatch')
    manifest=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  sources=source,checkpoint=str(checkpoint),checkpoint_sha256=config['checkpoint_sha256'],
                  attention_source=attention_module.__file__,attention_sha256=sha256(attention_module.__file__),
                  v8_attention_source=v8_attention_module.__file__,
                  v8_attention_sha256=sha256(v8_attention_module.__file__),
                  joint_alignment_source=joint_alignment_module.__file__,
                  joint_alignment_sha256=sha256(joint_alignment_module.__file__),
                  preflight=before,frame_ids=ids,windows=windows,preprocessing=saved['preprocessing'],input_sha256=sha256(args.input),
                  torch=torch.__version__,cuda=torch.version.cuda,gpu_uuid=before['gpu_uuid'],precision='bf16',
                  backend='PyTorch SDPA dispatcher on allowed-edge submatrices',
                  attention_backend_flags=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                                               memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                               math=torch.backends.cuda.math_sdp_enabled()),
                  cpu_offload=False,position_convention='OpenCV w2c extrinsics, stored c2w; intrinsics in resized-image pixels',
                  point_map='original point-head world_points for fixed Long alignment; depth-unprojection saved separately',seed=2026)
    manifest["attention_semantics"]=dict(mode=args.mode,
        local_kv="all local tokens once",remote_kv="one same-frame same-patch token in adjacent window",
        normalization="one joint local-plus-remote softmax; chunked explicit scores for correspondence queries")
    manifest["alignment_configuration"]=dict(
        mode=args.alignment_mode,
        direction="B_local -> A_local",
        ownership="front window first",
        huber_delta=0.1,
        lambda_center=1.0,
        lambda_rotation=1.0,
        gt_used_for_alignment=False)
    manifest["backend_profile"]=dict(requested=profile,effective=snapshot(torch),
        configured_before_cuda_initialization=True,head_frame_chunk=getattr(args,"dense_head_frame_chunk",None),
        historical_native_cublas_environment="unrecorded in 20260922T031101Z baseline")
    write_json(args.output/'run_manifest.json',manifest)
    model_start=time.perf_counter()
    model=VGGT().eval().requires_grad_(False)
    weights=load_file(str(checkpoint));model.load_state_dict(weights,strict=True);del weights
    model.cuda()
    if not whole_task:torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();model_seconds=time.perf_counter()-model_start
    from experiments.ours_v7.diagnostic_export import save_npz_fast
    save_npz=save_npz_fast if getattr(args,"npz_compression_level",6)==1 else np.savez_compressed
    if args.stream_independent:
        from experiments.ours_v8.stream_independent import execute_streaming
        execute_streaming(args,model,images,ids,windows,saved,save_npz,manifest,
                          task_start,input_seconds,model_seconds,profile)
        return
    start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        result=WindowReconstructor(model)(images,ids,mode=args.mode,window_size=args.window_size,overlap=args.overlap,
                                          query_chunk_size=args.query_chunk_size,
                                          reuse_image_encoding=args.reuse_image_encoding,
                                          cache_local_kv_dtype=args.cache_local_kv_dtype,
                                          correspondence_attention_path=args.correspondence_attention_path,
                                          dense_head_frame_chunk=getattr(args,'dense_head_frame_chunk',None))
    torch.cuda.synchronize();forward=time.perf_counter()-start
    manifest["correspondence"] = result["correspondence"]
    manifest["communication_mode"] = args.mode
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    del model
    if not whole_task:torch.cuda.empty_cache()
    export_start=time.perf_counter()
    predictions=[]
    for i,pred in enumerate(result['predictions']):
        values={k:v.numpy() if torch.is_tensor(v) else v for k,v in pred.items()}
        predictions.append(values)
        folder=args.output/'windows'/f'{i:04d}';folder.mkdir(parents=True,exist_ok=False)
        save_npz(folder/'local.npz',**values)
    export_seconds=time.perf_counter()-export_start
    from vggt.v8.joint_alignment import AlignmentStitcher,JointAlignmentConfig
    from experiments.ours_v4.artifacts import evaluate,write_cloud
    from experiments.ours_v6.correspondence import save_overlap_mask
    from experiments.ours_v6.metrics import summarize
    stitch=AlignmentStitcher(args.output/'alignment',JointAlignmentConfig(mode=args.alignment_mode))
    cloud=[];cloud_windows=[];stitch_seconds=0.;diagnostic_seconds=0.
    dense_keys=('c2w','intrinsics','depth','confidence','world_points','world_points_conf')
    owned_dense={key:[] for key in dense_keys}
    for i,(pred,(lo,hi)) in enumerate(zip(predictions,windows)):
        if i:
            save_overlap_mask(predictions[i-1],pred,
                args.output/'alignment'/f'edge_{i-1:04d}_{i:04d}_correspondence_mask.npz')
        start=time.perf_counter();fresh,global_pred=stitch.add(pred,i);stitch_seconds+=time.perf_counter()-start
        for key in dense_keys: owned_dense[key].append(global_pred[key][fresh])
        start=time.perf_counter();folder=args.output/'windows'/f'{i:04d}'
        xyz,_=write_point_cloud(folder/'point_head_global.ply',global_pred,images[lo:hi],fresh,i)
        write_cloud(folder/'depth_unprojection_global.ply',global_pred,images[lo:hi],fresh,16,i)
        cloud.append(xyz);cloud_windows.extend([i]*len(xyz));diagnostic_seconds+=time.perf_counter()-start
    global_result=stitch.finish(ids);save_npz(args.output/'global_trajectory.npz',**global_result)
    dense_export_start=time.perf_counter()
    final_dense={key:np.concatenate(chunks) for key,chunks in owned_dense.items()}
    if not np.array_equal(final_dense['c2w'],global_result['c2w']) or not np.array_equal(final_dense['intrinsics'],global_result['intrinsics']):
        raise ValueError('dense ownership disagrees with final trajectory')
    save_npz(args.output/'global_predictions.npz',
                        frame_ids=global_result['frame_ids'],
                        source_window=global_result['source_window'],**final_dense)
    export_seconds+=time.perf_counter()-dense_export_start
    start=time.perf_counter();evaluate(global_result,saved['scene_root'],args.output);summary=summarize(args.output);evaluation_seconds=time.perf_counter()-start
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
    if whole_task:
        allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    manifest["optimization"]=dict(backend_profile=profile,
        correspondence_query_chunk_size=args.query_chunk_size,
        correspondence_attention_path=args.correspondence_attention_path,
        reuse_image_encoding=args.reuse_image_encoding,
        dense_head_frame_chunk=getattr(args,"dense_head_frame_chunk",None),
        cache_local_kv_dtype=args.cache_local_kv_dtype,
        cache_sdpa_keys=False,exchange_flash_scope=False,
        npz_compression_level=getattr(args,"npz_compression_level",6),
        whole_task_peak_scope=whole_task,peak_resets=1 if whole_task else "legacy",
        deterministic=torch.are_deterministic_algorithms_enabled(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark)
    prep=saved['preprocessing']['elapsed_seconds']
    manifest.update(evaluation_summary=summary,timing=dict(result['timing'],image_preprocessing_seconds=prep,forward_seconds=forward,
                                input_read_seconds=input_seconds,model_load_seconds=model_seconds,
                                task_seconds=time.perf_counter()-task_start,
                                reconstruction_without_historical_preprocessing_seconds=forward+stitch_seconds,
                                overlap_stitching_seconds=stitch_seconds,reconstruction_total_seconds=prep+forward+stitch_seconds,
                                protocol_evaluation_seconds=evaluation_seconds,export_seconds=export_seconds,
                                diagnostic_seconds=diagnostic_seconds),
                    memory=result['memory'],retained_head_features_bytes=result['retained_head_features_bytes'],
                    peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                    cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,trainable_parameters=0)
    write_json(args.output/'run_manifest.json',manifest)
    write_json(args.output/'COMPLETE.json',dict(status='complete',kind='diagnostic',mode=args.mode))


def parse_args(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--frames',type=int,required=True)
    parser.add_argument('--window-size',type=int,default=60)
    parser.add_argument('--overlap',type=int,default=30)
    parser.add_argument('--mode',choices=['independent','overlap_correspondence'],required=True)
    parser.add_argument('--dense-head-frame-chunk',type=int,default=None)
    parser.add_argument('--npz-compression-level',type=int,choices=(1,6),default=6)
    parser.add_argument('--whole-task-measurement',action='store_true')
    parser.add_argument('--backend-profile',choices=('legacy','native_vggt'),default='legacy')
    parser.add_argument('--query-chunk-size',type=int,default=16,
                        help='Corresponding patch Queries per explicit joint-softmax chunk')
    parser.add_argument('--reuse-image-encoding',action='store_true')
    parser.add_argument('--cache-local-kv-dtype',action='store_true',
                        help='Cache converted local K/V only within one global layer and window')
    parser.add_argument('--correspondence-attention-path',
                        choices=('explicit','native_sdpa'),default='explicit',
                        help='Keep legacy explicit math or use bounded-mask native SDPA')
    parser.add_argument('--alignment-mode',
                        choices=('point_legacy','point_normalized_control','point_camera_joint'),
                        default='point_legacy',
                        help='Opt-in edge Sim(3) objective; legacy remains the default')
    parser.add_argument('--stream-independent',action='store_true',
                        help='Run and decode one independent window at a time')
    args=parser.parse_args(argv)
    if args.query_chunk_size < 1:
        parser.error('--query-chunk-size must be positive')
    if args.dense_head_frame_chunk is not None and args.dense_head_frame_chunk <= 0:
        parser.error('--dense-head-frame-chunk must be positive')
    if args.stream_independent and args.mode != 'independent':
        parser.error('--stream-independent requires --mode independent')
    if args.stream_independent and not args.whole_task_measurement:
        parser.error('--stream-independent requires --whole-task-measurement')
    if args.stream_independent and args.reuse_image_encoding:
        parser.error('--stream-independent does not support --reuse-image-encoding')
    return args

def main():
    args=parse_args()
    fresh_directory(args.output)
    try:execute(args)
    except Exception as error:
        write_json(args.output/'FAILED.json',dict(reason=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
