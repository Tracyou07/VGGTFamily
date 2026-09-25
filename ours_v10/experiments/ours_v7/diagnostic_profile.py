"""Serial GPU memory/timing instrumentation; output is never a performance claim."""
import argparse
from contextlib import AbstractContextManager
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.ours_v7.diagnostics import (ROOT,PARENT,OLD,WINDOWS,FIELDS,configure,
    Observer,inventory,storage_bytes,flatten,write_json,csv_rows,difference,load_ours,convert_native)

class RuntimeObserver(AbstractContextManager):
    def __init__(self,observer,out):
        self.obs=observer;self.out=out;self.patches=[];self.sdpa={};self.history=False
    def replace(self,module,name,replacement):
        old=getattr(module,name);self.patches.append((module,name,old));setattr(module,name,replacement);return old
    def __enter__(self):
        import torch.nn.functional as F
        original=F.scaled_dot_product_attention
        def sdpa(q,k,v,*args,**kwargs):
            key=str((tuple(q.shape),tuple(k.shape),str(q.dtype),str(k.dtype),kwargs.get('attn_mask',None) is not None))
            row=self.sdpa.setdefault(key,dict(q_shape=list(q.shape),k_shape=list(k.shape),v_shape=list(v.shape),
                dtype=str(q.dtype),calls=0,dense_mask=False,qkv_storage_bytes=storage_bytes([q,k,v]),
                max_allocated_at_call=0))
            row['calls']+=1
            row['max_allocated_at_call']=max(row['max_allocated_at_call'],torch.cuda.memory_allocated())
            if kwargs.get('attn_mask') is not None or args:row['dense_mask']=True
            return original(q,k,v,*args,**kwargs)
        self.replace(F,'scaled_dot_product_attention',sdpa)
        try:
            import vggt.v7.scheduler as scheduler
            import vggt.v7.attention as attention
            init=scheduler.initialize;step=scheduler.global_step
            def initialize(*args,**kwargs):
                self.obs.sample('window_initialization','before',args[1])
                result=init(*args,**kwargs)
                self.obs.sample('window_initialization','after',result)
                return result
            def global_step(*args,**kwargs):
                self.obs.sample('global_step','before',args[1])
                result=step(*args,**kwargs)
                self.obs.sample('global_step','after',result)
                return result
            self.replace(scheduler,'initialize',initialize)
            self.replace(scheduler,'global_step',global_step)
            bank=attention.communication_bank
            def communication_bank(*args,**kwargs):
                self.obs.sample('communication_bank','before')
                result=bank(*args,**kwargs)
                self.obs.sample('communication_bank','after',result)
                return result
            self.replace(attention,'communication_bank',communication_bank)
        except ModuleNotFoundError:
            pass
        torch.cuda.memory._record_memory_history(max_entries=100000)
        self.history=True
        def first_encoder(m,a,result):
            if not self.history:return
            snap=torch.cuda.memory._snapshot()
            allocations=[e for e in snap['device_traces'][0] if e['action']=='alloc']
            largest=sorted(allocations,key=lambda e:e['size'],reverse=True)[:25]
            write_json(self.out,dict(largest_allocations=largest,
                                    recorded_events=len(snap['device_traces'][0]),
                                    note='first image encoder allocation trace; no tensor contents'))
            torch.cuda.memory._record_memory_history(enabled=None);self.history=False
        self.handle=self.obs.model.aggregator.patch_embed.register_forward_hook(first_encoder)
        self.head_captured=False
        def before_head(m,a):
            if not self.head_captured:
                torch.cuda.memory._record_memory_history(max_entries=100000)
                self.history=True
        def after_head(m,a,result):
            if self.head_captured:return
            snap=torch.cuda.memory._snapshot()
            entries=snap['device_traces'][0]
            allocations=[e for e in entries if e['action']=='alloc']
            write_json(self.out.with_name(self.out.stem+'_depth_head.json'),
                dict(largest_allocations=sorted(allocations,key=lambda e:e['size'],reverse=True)[:20],
                     peak_allocated=torch.cuda.max_memory_allocated(),
                     active_after=torch.cuda.memory_allocated(),
                     note='first depth head allocation trace; no activation contents'))
            torch.cuda.memory._record_memory_history(enabled=None)
            self.history=False;self.head_captured=True
        self.head_pre=self.obs.model.depth_head.register_forward_pre_hook(before_head)
        self.head_post=self.obs.model.depth_head.register_forward_hook(after_head)
        return self
    def __exit__(self,*args):
        self.handle.remove();self.head_pre.remove();self.head_post.remove()
        if self.history:torch.cuda.memory._record_memory_history(enabled=None)
        for module,name,old in reversed(self.patches):setattr(module,name,old)
        return False

def run(out,mode,repeats,query_chunk_size=64,cache_keys=False):
    label=mode if query_chunk_size==64 else f"{mode}_q{query_chunk_size}"
    if mode in ('full','windows'):
        sys.path.insert(0,str(PARENT/'vggt'))
    if cache_keys:label+="_cached_keys"
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    configure()
    t=time.perf_counter();saved=torch.load(OLD/'inputs.pt',map_location='cpu',weights_only=True)
    input_load=time.perf_counter()-t
    manifest=json.loads((OLD/'independent/run_manifest.json').read_text())
    t=time.perf_counter();model=VGGT().eval().requires_grad_(False)
    status=model.load_state_dict(load_file(manifest['checkpoint']),strict=True);model.cuda()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-t
    baseline=torch.cuda.memory_allocated()
    write_json(out/f'{label}_load.json',dict(model_load_seconds=load_seconds,input_file_load_seconds=input_load,
        preprocessing_seconds=0,preprocessing='reuse original tensor; no decode/resize',baseline_allocated=baseline,
        baseline_reserved=torch.cuda.memory_reserved(),parameters=inventory(list(model.parameters())),
        model_unique_storage_bytes=storage_bytes(list(model.parameters())+list(model.buffers())),
        model_training=model.training,strict_load=str(status),grad_parameters=sum(p.requires_grad for p in model.parameters()),
        module_source=sys.modules[VGGT.__module__].__file__,deterministic=torch.are_deterministic_algorithms_enabled(),
        cudnn_tf32=torch.backends.cudnn.allow_tf32))
    timing=[];mem=[];check=[]
    for iteration in range(repeats+1):
        diagnostic=iteration==repeats
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        observer=Observer(model,enabled=diagnostic,label=label)
        observer.sample('baseline','before')
        from contextlib import nullcontext
        from experiments.ours_v7.diagnostic_key_cache import cached_sdpa_keys
        key_context=cached_sdpa_keys() if cache_keys else nullcontext({})
        instrument=RuntimeObserver(observer,out/f'{label}_allocation_trace.json') if diagnostic else None
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),observer,key_context as key_stats:
            if instrument:instrument.__enter__()
            try:
                if mode in ('full','windows'):
                    ranges=[(0,100)] if mode=='full' else WINDOWS
                    native=[]
                    for w,(lo,hi) in enumerate(ranges):
                        observer.sample('input_transfer','before')
                        x=saved['images'][lo:hi].cuda()
                        observer.sample('input_transfer','after',x)
                        raw=model(x)
                        values=convert_native(raw,x.shape[-2:])
                        native.append(values)
                        del raw,x
                    result={'predictions':native}
                else:
                    from vggt.v7.model import WindowReconstructor
                    result=WindowReconstructor(model)(saved['images'],saved['frame_ids'],
                        mode='independent' if mode=='independent' else 'camera_patch_exchange',
                        window_size=60,overlap=30,query_chunk_size=query_chunk_size,patch_exchange_ratio=.1)
            finally:
                if instrument:instrument.__exit__(None,None,None)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        timing.append(dict(run=label,iteration=iteration,diagnostic=diagnostic,cold=iteration==0,
                           stage='forward_and_cpu_output_transfer',seconds=elapsed,compiled=False,
                           **result.get('timing',{})))
        if cache_keys:write_json(out/f'{label}_cache_stats_{iteration}.json',key_stats)
        peak=torch.cuda.max_memory_allocated()
        mem.append(dict(run=label,iteration=iteration,diagnostic=diagnostic,
            stage='entire_forward',allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
            peak_allocated=peak,peak_reserved=torch.cuda.max_memory_reserved(),baseline_allocated=baseline))
        if diagnostic:
            csv_rows(out/f'{label}_stages.csv',observer.rows)
            write_json(out/f'{label}_tensor_inventory.json',observer.tensors)
            write_json(out/f'{label}_sdpa.json',list(instrument.sdpa.values()))
            if 'memory' in result:write_json(out/f'{label}_scheduler_memory.json',result['memory'])
        for w,pred in enumerate(result['predictions']):
            if mode=='full':
                prior=np.load(Path('/data/yjh/output/vggt/fixed_input_baselines/20260922T031101Z_fixed_baselines_vggt_star/global_trajectory.npz'))['c2w']
                fields=['c2w'];reference={'c2w':prior}
                np.savez(out/f'{label}_trajectory_{iteration}.npz',frame_ids=saved['frame_ids'],c2w=pred['c2w'])
            elif mode=='patch':
                p=Path('/data/yjh/output/vggt/ours_v7/20260922T023028Z_v7_f100_camera_patch_exchange/camera_patch_exchange/windows')/f'{w:04d}'/'local.npz'
                with np.load(p) as z:reference={k:z[k] for k in FIELDS}
                fields=FIELDS
            else:
                reference=load_ours(w);fields=FIELDS
            for key in fields:
                value=pred[key].numpy() if torch.is_tensor(pred[key]) else pred[key]
                check.append(dict(run=label,iteration=iteration,diagnostic=diagnostic,window=w,field=key,
                                  **difference(reference[key],value)))
        del result
        print(mode,iteration,'diagnostic',diagnostic,'seconds',elapsed,'peak_GiB',peak/2**30,flush=True)
        csv_rows(out/f'{label}_timing.csv',timing);csv_rows(out/f'{label}_memory.csv',mem)
        csv_rows(out/f'{label}_regression.csv',check)
    del model;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['full','windows','independent','patch'],required=True)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--query-chunk-size',type=int,default=64)
    p.add_argument('--cache-keys',action='store_true')
    a=p.parse_args();run(a.output,a.mode,a.repeats,a.query_chunk_size,a.cache_keys)
