"""Read-only, 61-frame real-block and full-backbone communication probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from vggt.models.vggt import VGGT
from vggt.v6 import attention as v6_attention
from vggt.v6.attention import global_step, exchange_attention, communication_bank, layout
from vggt.v6.scheduler import initialize, aggregate_windows
from experiments.ours_v6.runtime import preflight, sha256, write_json
from experiments.ours_v6.windows import make_windows

MODES=('independent','camera_exchange','camera_register_exchange')


def difference(a,b):
    d=(a.float()-b.float()).abs()
    return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),l2=float(d.norm()))


def slices(t,per_frame,camera,register):
    frames=t.shape[1]//per_frame
    shaped=t.reshape(1,frames,per_frame,t.shape[-1])
    return dict(camera=shaped[:,:,0:camera],register=shaped[:,:,camera:camera+register],
                patch=shaped[:,:,camera+register:])


def summarize_pair(a,b,per_frame,camera,register):
    out={k:difference(slices(a,per_frame,camera,register)[k],
                      slices(b,per_frame,camera,register)[k])
         for k in ('camera','register','patch')}
    out['whole_block']=difference(a,b)
    return out


def attention_output(block,state,pos,per_frame,bank,index,count):
    normalized=block.norm1(state)
    if bank is None:
        return block.attn(normalized,pos=pos)
    return exchange_attention(block.attn,normalized,pos,per_frame,bank,index,count)


def single_block_probe(aggregator,images,windows):
    states,positions,per_frame=initialize(aggregator,images,windows)
    camera,register=layout(aggregator)
    channels=states[0].shape[-1]
    for i,(lo,hi) in enumerate(windows):
        frames=hi-lo
        pos=None if positions[i] is None else positions[i].reshape(frames,per_frame,2)
        states[i]=aggregator.frame_blocks[0](states[i].reshape(frames,per_frame,channels),pos=pos).reshape(1,-1,channels)
    block=aggregator.global_blocks[0]
    outputs={};attention_outputs={};records={}
    for mode in MODES:
        count=camera if mode=='camera_exchange' else camera+register
        bank=None if mode=='independent' else [communication_bank(block,x,p,per_frame,count)
                                                    for x,p in zip(states,positions)]
        attention_outputs[mode]=[attention_output(block,s,p,per_frame,bank,i,count)
                                  for i,(s,p) in enumerate(zip(states,positions))]
        rows=[]
        outputs[mode]=global_step(block,states,positions,per_frame,mode,camera,register,
                                  diagnostics=rows,layer=0)
        records[mode]=rows
        if bank is not None:
            zeroed=[(torch.zeros_like(k),torch.zeros_like(v)) for k,v in bank]
            attention_outputs[mode+'_remote_bank_zeroed']=[attention_output(block,s,p,per_frame,zeroed,i,count)
                                            for i,(s,p) in enumerate(zip(states,positions))]
            original=v6_attention.communication_bank
            def make_zeroed(*args,**kwargs):
                k,v=original(*args,**kwargs)
                return torch.zeros_like(k),torch.zeros_like(v)
            with patch.object(v6_attention,'communication_bank',side_effect=make_zeroed):
                outputs[mode+'_remote_bank_zeroed']=global_step(block,states,positions,per_frame,
                       mode,camera,register)
    comparisons={}
    pairs=(('camera_exchange','independent'),
           ('camera_register_exchange','camera_exchange'),
           ('camera_exchange','camera_exchange_remote_bank_zeroed'),
           ('camera_register_exchange','camera_register_exchange_remote_bank_zeroed'))
    for a,b in pairs:
        comparisons[a+'__vs__'+b]=[dict(attention=summarize_pair(aa,bb,per_frame,camera,register),
                                         block=summarize_pair(ab,bb2,per_frame,camera,register))
                                      for aa,bb,ab,bb2 in zip(attention_outputs[a],attention_outputs[b],outputs[a],outputs[b])]
    return outputs,dict(layer=0,module='aggregator.global_blocks.0',windows=windows,
                        token_counts=dict(camera=camera,register=register,patch_per_frame=per_frame-camera-register),
                        attention_shapes=[list(s.shape) for s in states],comparisons=comparisons,
                        actual_attention_rows=records)


def register_probe(aggregator,outputs,windows,height,width):
    per_frame=outputs['camera_exchange'][0].shape[1]//(windows[0][1]-windows[0][0])
    camera,register=layout(aggregator)
    next_block=aggregator.frame_blocks[1]
    rows=[]
    for i,(lo,hi) in enumerate(windows):
        source=outputs['camera_exchange'][i]
        target=outputs['camera_register_exchange'][i]
        before=slices(source,per_frame,camera,register)['register']
        after=slices(target,per_frame,camera,register)['register']
        pos=aggregator.position_getter(hi-lo,height//aggregator.patch_size,width//aggregator.patch_size,device=source.device)+1
        special=torch.zeros(hi-lo,aggregator.patch_start_idx,2,device=source.device,dtype=pos.dtype)
        pos=torch.cat((special,pos),dim=1)
        seen=[]
        handle=next_block.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
        try:
            next_source=next_block(source.reshape(hi-lo,per_frame,-1),pos=pos).reshape(1,-1,source.shape[-1])
            next_target=next_block(target.reshape(hi-lo,per_frame,-1),pos=pos).reshape(1,-1,target.shape[-1])
        finally:
            handle.remove()
        writeback_exact=(len(seen)==2 and torch.equal(seen[0].reshape_as(source),source)
                         and torch.equal(seen[1].reshape_as(target),target))
        rows.append(dict(window=i,global_block_input_to_next_frame_input_exact=writeback_exact,
                         register_camera_exchange_norm=float(before.float().norm(dim=-1).mean()),
                         register_camera_register_exchange_norm=float(after.float().norm(dim=-1).mean()),
                         register_remote_delta=difference(before,after),
                         next_frame_block_register_delta=difference(
                             slices(next_source,per_frame,camera,register)['register'],
                             slices(next_target,per_frame,camera,register)['register'])))
    return dict(layer=0,next_frame_layer=1,rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=61)
    args=parser.parse_args()
    if args.frames!=61: raise ValueError('fixed real-block probe requires 61 frames')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu:
        raise ValueError('CUDA_VISIBLE_DEVICES must match --gpu')
    pre=preflight(args.gpu,args.output.parent)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config=json.loads((Path(__file__).resolve().parents[2]/'configs/v6_validation.json').read_text())
    if sha256(config['checkpoint'])!=config['checkpoint_sha256']:
        raise ValueError('checkpoint checksum mismatch')
    saved=torch.load(args.input,map_location='cpu',weights_only=True)
    images=saved['images'][:61];frame_ids=saved['frame_ids'][:61]
    if len(images)!=61 or len(frame_ids)!=61: raise ValueError('input lacks 61 fixed frames')
    windows=make_windows(61,60,30)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    model=VGGT().eval().requires_grad_(False)
    model.load_state_dict(load_file(config['checkpoint']),strict=True)
    model.cuda()
    metadata=dict(preflight=pre,frames=61,frame_ids=frame_ids,windows=windows,
                  input=str(args.input),input_sha256=sha256(args.input),checkpoint=config['checkpoint'],
                  checkpoint_sha256=config['checkpoint_sha256'],precision='bf16',seed=2026,
                  diagnostic_weights='FP32 softmax recomputed from the actual projected Q/K; production outputs use SDPA',
                  sampled_query_frames_per_window=4)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        outputs,attention_probe=single_block_probe(model.aggregator,images,windows)
        write_json(args.output/'attention_probe.json',attention_probe)
        write_json(args.output/'register_probe.json',register_probe(model.aggregator,outputs,windows,images.shape[-2],images.shape[-1]))
        del outputs
        for mode in MODES:
            rows=[]
            torch.cuda.synchronize();start=time.perf_counter()
            features,_,memory=aggregate_windows(model.aggregator,images,windows,mode,
                                                  diagnostics=rows,frame_ids=frame_ids)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            del features
            folder=args.output/mode;folder.mkdir()
            write_json(folder/'communication_diagnostics.json',dict(**metadata,mode=mode,
                        global_blocks=model.aggregator.depth,rows=rows,
                        memory=memory,elapsed_seconds=elapsed,
                        note='Patch delta is zero by allowed-edge topology at this attention call. Query deltas compare sampled remote enabled versus local-only attention on identical block input.'))
    write_json(args.output/'probe_manifest.json',metadata)
    write_json(args.output/'COMPLETE.json',dict(status='complete',kind='v6_communication_probe',frames=61))

if __name__=='__main__':main()
