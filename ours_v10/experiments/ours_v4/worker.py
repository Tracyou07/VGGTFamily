"""Separate process: reference selects the immutable original package first."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
if '--reference' in sys.argv:
    sys.path.insert(0,str(ROOT/'reference'))
else:
    sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from experiments.ours_v4.core import pack_groups
from experiments.ours_v3.geometry import AlignmentConfig
from experiments.ours_v3.stitch import Stitcher,json_write
from experiments.ours_v4.artifacts import evaluate,write_cloud
# overlap helper is not part of reference vggt: define the exact window rule locally.


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',action='store_true')
    p.add_argument('--capture',action='store_true')
    p.add_argument('--precision',choices=['fp32','bf16'],required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--window-batch-size',type=int,default=2)
    a=p.parse_args()
    output=Path(a.output); output.mkdir(parents=True,exist_ok=False)
    start_all=time.perf_counter()
    try:
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        import vggt.layers.attention as attn_module
        torch.manual_seed(2026); np.random.seed(2026)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        saved=torch.load(a.input,map_location='cpu',weights_only=True)
        images=saved['images'][:a.frames]; ids=saved['frame_ids'][:a.frames]
        if len(ids)!=a.frames: raise ValueError('insufficient frames')
        windows=[(s,min(s+30,a.frames)) for s in range(0,a.frames,20)]
        groups=[[i] for i in range(len(windows))] if a.reference else pack_groups(windows,a.window_batch_size)
        checkpoint='/data/yjh/share/pretrained/VGGT-1B/model.safetensors'
        digest=hashlib.sha256()
        with open(checkpoint,'rb') as f:
            for block in iter(lambda:f.read(1<<20),b''): digest.update(block)
        expected='f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
        if digest.hexdigest()!=expected: raise ValueError('checkpoint hash mismatch')
        model=VGGT().eval().requires_grad_(False)
        weights=load_file(checkpoint); model.load_state_dict(weights,strict=True); del weights
        model.cuda()
        captured={}
        def embed_hook(module,args,value):
            captured['encoder']=value['x_norm_patchtokens'] if isinstance(value,dict) else value
        def aggregate_hook(module,args,value):
            captured['features']=value[0]
        handles=[]
        if a.capture:
            handles=[model.aggregator.patch_embed.register_forward_hook(embed_hook),
                     model.aggregator.register_forward_hook(aggregate_hook)]
        manifest=dict(reference=a.reference,reference_commit='cc1d8ac15861aea54d14961653cd340e7d984f29',
            v4_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            attention_source=str(Path(attn_module.__file__).resolve()),
            attention_sha256=hashlib.sha256(Path(attn_module.__file__).read_bytes()).hexdigest(),
            checkpoint=checkpoint,checkpoint_sha256=expected,precision=a.precision,seed=2026,
            input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
            frame_ids=ids,windows=windows,groups=groups,window_batch_size=1 if a.reference else a.window_batch_size,
            python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            attention_backend='unmodified PyTorch SDPA; dispatcher selected kernel',
            backend_flags=dict(flash=torch.backends.cuda.flash_sdp_enabled(),memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled()),
            gpu=torch.cuda.get_device_name(),tf32=False,deterministic_algorithms=True,
            model_config=dict(img_size=518,patch_size=14,embed_dim=1024,heads='original camera/depth/point'),
            preprocessing=saved['preprocessing'],feature_cache=False)
        json_write(output/'manifest.json',manifest)
        torch.cuda.reset_peak_memory_stats(); seconds=0.
        predictions={}
        for group in groups:
            batch=torch.stack([images[windows[i][0]:windows[i][1]].clone() for i in group]).cuda()
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
                out=model(batch)
            torch.cuda.synchronize(); seconds+=time.perf_counter()-start
            with torch.inference_mode():
                ext,intr=pose_encoding_to_extri_intri(out['pose_enc'].float(),image_size_hw=batch.shape[-2:])
                bottom=torch.zeros((*ext.shape[:2],1,4),device=ext.device); bottom[...,0,3]=1
                c2w=torch.linalg.inv(torch.cat([ext,bottom],-2))
            encoder=None
            if a.capture:
                encoder=captured['encoder'].reshape(len(group),batch.shape[1],*captured['encoder'].shape[1:])
            for offset,index in enumerate(group):
                lo,hi=windows[index]
                values=dict(pose_encoding=out['pose_enc'][offset].float().cpu().clone(),
                            depth=out['depth'][offset].float().cpu().clone(),
                            confidence=out['depth_conf'][offset].float().cpu().clone(),
                            c2w=c2w[offset].float().cpu().clone(),intrinsics=intr[offset].float().cpu().clone())
                if a.capture:
                    values['encoder']=encoder[offset].cpu().clone()
                    for layer,tensor in enumerate(captured['features']):
                        if tensor is not None: values[f'head_cache_{layer}']=tensor[offset].cpu().clone()
                    torch.save(values,output/f'window_{index:04d}.pt')
                    del tensor
                prediction={key:values[key].numpy() for key in ('pose_encoding','depth','confidence','c2w','intrinsics')}
                prediction['frame_ids']=ids[lo:hi]
                folder=output/'windows'/f'{index:04d}'; folder.mkdir(parents=True)
                np.savez_compressed(folder/'local.npz',**prediction)
                predictions[index]=prediction
            captured.clear()
            del out,batch,c2w,ext,intr,encoder,values
        for handle in handles: handle.remove()
        alignment_seconds=0.
        if not a.capture:
            stitch=Stitcher(output/'alignment',AlignmentConfig())
            for index,(lo,hi) in enumerate(windows):
                prediction=predictions[index]
                start=time.perf_counter(); fresh,pose,depth=stitch.add(prediction,index)
                alignment_seconds+=time.perf_counter()-start
                folder=output/'windows'/f'{index:04d}'
                transformed={**prediction,'c2w':pose,'depth':depth}
                np.savez_compressed(folder/'global_new_frames.npz',frame_ids=np.asarray(prediction['frame_ids'])[fresh],
                    c2w=pose[fresh],depth=depth[fresh],intrinsics=prediction['intrinsics'][fresh],confidence=prediction['confidence'][fresh])
                write_cloud(folder/'global_new_frames.ply',transformed,images[lo:hi],fresh,16,index)
            result=stitch.finish(ids); np.savez_compressed(output/'global_trajectory.npz',**result)
            evaluate(result,saved['scene_root'],output)
        manifest.update(inference_seconds=seconds,alignment_seconds=alignment_seconds,
            total_seconds=time.perf_counter()-start_all,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),alignment_config=AlignmentConfig().__dict__)
        json_write(output/'manifest.json',manifest); json_write(output/'COMPLETE.json',dict(status='complete'))
    except Exception as e:
        json_write(output/'FAILED.json',dict(reason=str(e),traceback=traceback.format_exc()))
        raise

if __name__=='__main__': main()
