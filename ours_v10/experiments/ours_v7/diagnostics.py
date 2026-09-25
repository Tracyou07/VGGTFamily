"""Opt-in v7 diagnostics. Never imported by production inference."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from contextlib import AbstractContextManager
import numpy as np
import torch

ATOL = 2e-5
RTOL = 2e-5  # Existing tests/ours_v7/test_model.py contract, fixed before execution.
ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent
OLD = Path('/data/yjh/output/vggt/ours_v7/20260922T023028Z_v7_f100_independent')
LONG = Path('/data/yjh/output/vggt/fixed_input_baselines/20260922T031101Z_fixed_baselines_vggt_long')
STAR = Path('/data/yjh/output/vggt/fixed_input_baselines/20260922T031101Z_fixed_baselines_vggt_star')
WINDOWS = [(0,60),(30,90),(60,100)]
FIELDS = ['pose_encoding','c2w','intrinsics','depth','depth_conf','world_points','world_points_conf']

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str)+'\n')

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def tensor_hash(value):
    if torch.is_tensor(value):
        value=value.detach().cpu().contiguous()
        value=value.view(torch.uint8).numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def difference(reference, actual):
    a=np.asarray(reference);b=np.asarray(actual)
    if a.shape!=b.shape: raise ValueError((a.shape,b.shape))
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return dict(max_abs=None,mean_abs=None,relative_l2=None,finite=False,exact=False,within_tolerance=False,shape=list(a.shape))
    maximum=total=norm=base=0.;close=True
    av=a.reshape(-1);bv=b.reshape(-1)
    for i in range(0,av.size,1000000):
        x=av[i:i+1000000].astype(np.float64);y=bv[i:i+1000000].astype(np.float64)
        z=np.abs(x-y)
        maximum=max(maximum,float(z.max(initial=0)))
        total+=float(z.sum());norm+=float(z@z);base+=float(x@x)
        close=close and bool(np.all(z<=ATOL+RTOL*np.abs(x)))
    return dict(max_abs=maximum,mean_abs=total/max(av.size,1),
                relative_l2=float(np.sqrt(norm/max(base,1e-300))),finite=finite,
                exact=bool(np.array_equal(a,b)),within_tolerance=close,shape=list(a.shape))

def csv_rows(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def storage_bytes(tensors):
    stores={}
    for t in tensors:
        if torch.is_tensor(t):
            s=t.untyped_storage();stores[(str(t.device),s.data_ptr())]=s.nbytes()
    return sum(stores.values())

def flatten(value):
    if torch.is_tensor(value):return [value]
    if isinstance(value,dict):return [t for v in value.values() for t in flatten(v)]
    if isinstance(value,(tuple,list)):return [t for v in value for t in flatten(v)]
    return []

def inventory(values):
    seen=set();out=[]
    for t in flatten(values):
        s=t.untyped_storage();key=(str(t.device),s.data_ptr())
        out.append(dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),
                        logical_bytes=t.numel()*t.element_size(),storage_bytes=s.nbytes(),
                        first_storage_occurrence=key not in seen,requires_grad=t.requires_grad,
                        has_grad_fn=t.grad_fn is not None))
        seen.add(key)
    return out

class Observer(AbstractContextManager):
    """Read-only synchronized hooks. Timings explicitly include diagnostic overhead."""
    def __init__(self,model,enabled=True,cuda=True,label=''):
        self.model=model;self.enabled=enabled;self.cuda=cuda;self.label=label
        self.rows=[];self.handles=[];self.starts={};self.tensors=[]
    def sample(self,stage,event,values=None):
        if self.cuda:torch.cuda.synchronize()
        row=dict(run=self.label,stage=stage,event=event,wall=time.perf_counter(),
                 allocated=torch.cuda.memory_allocated() if self.cuda else 0,
                 reserved=torch.cuda.memory_reserved() if self.cuda else 0,
                 peak_allocated=torch.cuda.max_memory_allocated() if self.cuda else 0,
                 peak_reserved=torch.cuda.max_memory_reserved() if self.cuda else 0,
                 cpu_rss=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),
                 grad_enabled=torch.is_grad_enabled(),diagnostic=True)
        if event=='before':self.starts[stage]=row['wall']
        if event=='after':row['seconds']=row['wall']-self.starts.get(stage,row['wall'])
        if values is not None:
            row['output_unique_storage_bytes']=storage_bytes(flatten(values))
            self.tensors.append(dict(stage=stage,event=event,tensors=inventory(values)))
        self.rows.append(row)
    def __enter__(self):
        if not self.enabled:return self
        for name,module in self.model.named_modules():
            observe=(name in ('aggregator','aggregator.patch_embed','camera_head','depth_head','point_head')
                     or (name.startswith(('aggregator.frame_blocks.','aggregator.global_blocks.')) and name.count('.')==2)
                     or (name.startswith('aggregator.patch_embed.blocks.') and name.count('.')==3)
                     or name in ('depth_head.scratch.refinenet1','point_head.scratch.refinenet1',
                                 'depth_head.scratch.output_conv1','point_head.scratch.output_conv1',
                                 'depth_head.scratch.output_conv2','point_head.scratch.output_conv2'))
            if observe:
                self.handles.append(module.register_forward_pre_hook(lambda m,a,n=name:self.sample(n,'before')))
                self.handles.append(module.register_forward_hook(lambda m,a,o,n=name:self.sample(n,'after',o)))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        return False

def assemble(predictions,transforms,ownership,path):
    """Two independently expressed formulas from ours transform_predictions and Long save_camera_poses."""
    from experiments.ours_v3.geometry import transform_predictions
    poses={};owners={}
    for w,(pred,T) in enumerate(zip(predictions,transforms)):
        if path=='ours':
            p,_=transform_predictions(pred['c2w'],np.ones(1),T)
        elif path=='long':
            p=np.array(pred['c2w'],copy=True)
            p[:,:3,:3]=np.einsum('ij,bjk->bik',T.rotation,p[:,:3,:3])
            p[:,:3,3]=(T.scale*(T.rotation@p[:,:3,3].T)).T+T.translation
        else:raise ValueError(path)
        for k,f in enumerate(pred['frame_ids']):
            if ownership=='last' or f not in poses:poses[f]=p[k];owners[f]=w
    ids=sorted(poses)
    return np.stack([poses[f] for f in ids]),np.array([owners[f] for f in ids])

def load_ours(w):
    with np.load(OLD/'independent/windows'/f'{w:04d}'/'local.npz') as p:
        return {k:p[k] for k in p.files}

def load_native_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def audit(out):
    manifest=json.loads((OLD/'independent/run_manifest.json').read_text())
    long=json.loads((LONG/'long_manifest.json').read_text())
    saved=torch.load(OLD/'inputs.pt',weights_only=True,map_location='cpu')
    identities={}
    for name in ['ours_v7','vggt','fixed_input_baselines']:
        repo=PARENT/name
        identities[name]=dict(commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
            status=subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True).strip())
    if identities['vggt']['commit']!='cc1d8ac15861aea54d14961653cd340e7d984f29':
        raise ValueError('native source revision drift')
    if identities['fixed_input_baselines']['commit']!='d80946cfb1463ba82ec63530b016d87f63a9cf4d':
        raise ValueError('baseline adapter revision drift')
    checks=[]
    for rel,expected in manifest['sources']['sha256'].items():
        p=ROOT/rel
        checks.append(dict(source='ours_v7',file=str(p),expected=expected,actual=sha256(p),matches=sha256(p)==expected))
    for filename,expected in long['source_file_sha256'].items():
        checks.append(dict(source='long',file=filename,expected=expected,actual=sha256(filename),matches=sha256(filename)==expected))
    same=[]
    for folder in ['models','heads','layers','utils']:
        for p in (PARENT/'vggt/vggt'/folder).rglob('*.py'):
            rel=p.relative_to(PARENT/'vggt');ours=ROOT/rel
            same.append(dict(file=str(rel),equal=ours.is_file() and sha256(ours)==sha256(p)))
    weight=sha256(manifest['checkpoint'])
    receipt=json.loads((STAR/'input_receipt.json').read_text())
    record=dict(identities=identities,source_checks=checks,native_source_comparison=same,
        checkpoint=manifest['checkpoint'],checkpoint_sha256=weight,checkpoint_match=weight==manifest['checkpoint_sha256'],
        input_file_sha256=sha256(OLD/'inputs.pt'),input_manifest_match=sha256(OLD/'inputs.pt')==manifest['input_sha256'],
        images_sha256=tensor_hash(saved['images']),images_receipt_match=tensor_hash(saved['images'])==receipt['images_sha256'],
        windows=[dict(range=[lo,hi],frame_ids=saved['frame_ids'][lo:hi],tensor_sha256=tensor_hash(saved['images'][lo:hi]),
                      shape=list(saved['images'][lo:hi].shape),dtype=str(saved['images'].dtype)) for lo,hi in WINDOWS],
        preprocessing=saved['preprocessing'],tolerance=dict(atol=ATOL,rtol=RTOL,source='tests/ours_v7/test_model.py'),
        torch=torch.__version__,cuda=torch.version.cuda,
        native_first_reference='slice_expand_and_flatten index0 per window; index1 remaining',
        initialization='same checkpoint camera/register; first frame per window',
        rope='same patch-center grid +1; special tokens 0; no frame renumbering',
        baseline_adapter_manifest_commit=receipt['adapter_commit'])
    write_json(out/'source_manifest.json',record)
    assert weight==manifest['checkpoint_sha256']
    permitted_test=str(ROOT/'tests/ours_v7/test_attention.py')
    assert all(x['matches'] or x['file']==permitted_test for x in checks)
    # The only edited pre-existing file adds diagnostic query-tiling tests.
    # Baseline inference source bytes must still match the historical manifest.

    assert record['input_manifest_match'] and record['images_receipt_match']
    rows=[]
    for w,(lo,hi) in enumerate(WINDOWS):
        a=load_ours(w);b=np.load(LONG/'native_long/_tmp_results_unaligned'/f'chunk_{w}.npy',allow_pickle=True).item()
        rows.append(dict(comparison='input_vs_long_stored',window=w,field='images',**difference(saved['images'][lo:hi].numpy(),b['images'])))
        for key,bkey in [('intrinsics','intrinsic'),('depth','depth'),('depth_conf','depth_conf'),('world_points','world_points'),('world_points_conf','world_points_conf')]:
            ref=b[bkey]
            if key=='depth':ref=ref[...,None]
            rows.append(dict(comparison='stored_long_vs_ours',window=w,field=key,**difference(ref,a[key])))
        rows.append(dict(comparison='stored_long_vs_ours',window=w,field='c2w',**difference(b['extrinsic'],a['c2w'])))
        assert list(a['frame_ids'])==saved['frame_ids'][lo:hi]
        print('stored_window',w,[(x['field'],x['max_abs']) for x in rows if x['window']==w],flush=True)
    csv_rows(out/'stored_prediction_differences.csv',rows)
    write_json(out/'audit_summary.json',dict(all_manifest_hashes_match=all(x['matches'] for x in checks),all_inference_hashes_match=True,
        unequal_native_files=[x for x in same if not x['equal']],stored_rows=rows))

def configure():
    torch.manual_seed(2026);np.random.seed(2026);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)

def convert_native(pred,shape):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    with torch.autocast('cuda',enabled=False):
        e,k=pose_encoding_to_extri_intri(pred['pose_enc'].float(),image_size_hw=shape)
        bottom=torch.zeros((*e.shape[:2],1,4),device=e.device,dtype=e.dtype);bottom[...,0,3]=1
        c=torch.linalg.inv(torch.cat([e,bottom],dim=-2))
    values=dict(pose_encoding=pred['pose_enc'],c2w=c,intrinsics=k,
                **{key:pred[key] for key in FIELDS if key not in ('pose_encoding','c2w','intrinsics')})
    return {k:v[0].float().cpu().numpy() for k,v in values.items()}

def native_check(out, runtime_default=False):
    sys.path.insert(0,str(PARENT/'vggt'))
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    configure()
    if runtime_default:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.allow_tf32=True
    prefix='default_' if runtime_default else ''
    manifest=json.loads((OLD/'independent/run_manifest.json').read_text())
    saved=torch.load(OLD/'inputs.pt',map_location='cpu',weights_only=True)
    model=VGGT().eval().requires_grad_(False)
    status=model.load_state_dict(load_file(manifest['checkpoint']),strict=True)
    model.cuda()
    rows=[];time_rows=[];memory=[]
    for w,(lo,hi) in enumerate(WINDOWS):
        ours=load_ours(w);repeated=None
        long_raw=np.load(LONG/'native_long/_tmp_results_unaligned'/f'chunk_{w}.npy',allow_pickle=True).item() if runtime_default else None
        for repeat in range(2 if not runtime_default or w==0 else 1):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                gpu_input=saved['images'][lo:hi].cuda()
                raw=model(gpu_input)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                native=convert_native(raw,saved['images'].shape[-2:])
                del raw,gpu_input
            time_rows.append(dict(run='native_window',window=w,repeat=repeat,seconds=elapsed,scope='input_transfer_and_forward',diagnostic=False))
            memory.append(dict(run='native_window',window=w,repeat=repeat,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
            if repeat==0:
                repeated=native
            else:
                for key in FIELDS:
                    rows.append(dict(comparison='native_repeat',window=w,field=key,**difference(repeated[key],native[key])))
            for key in FIELDS:
                rows.append(dict(comparison='native_vs_stored_ours',window=w,repeat=repeat,field=key,**difference(native[key],ours[key])))
            if runtime_default:
                for key,lk in [('c2w','extrinsic'),('intrinsics','intrinsic'),('depth','depth'),('depth_conf','depth_conf'),('world_points','world_points'),('world_points_conf','world_points_conf')]:
                    a=long_raw[lk]
                    if key=='depth':a=a[...,None]
                    rows.append(dict(comparison='default_native_vs_stored_long',window=w,repeat=repeat,field=key,**difference(a,native[key])))
            print('native',w,repeat,elapsed,[(x['field'],x['max_abs']) for x in rows if x['comparison']=='native_vs_stored_ours' and x['window']==w and x['repeat']==repeat],flush=True)
            csv_rows(out/(prefix+'raw_prediction_differences.csv'),rows)
            csv_rows(out/(prefix+'native_window_timing.csv'),time_rows)
            csv_rows(out/(prefix+'native_window_memory.csv'),memory)
    write_json(out/(prefix+'native_check.json'),dict(strict_load=str(status),source_file=sys.modules[VGGT.__module__].__file__,
        native_vs_ours_all_exact=all(x['exact'] for x in rows if x['comparison']=='native_vs_stored_ours'),
        native_repeat_all_exact=all(x['exact'] for x in rows if x['comparison']=='native_repeat')))
    del model;torch.cuda.empty_cache()


def initialization_audit(out):
    from types import SimpleNamespace
    from safetensors import safe_open
    from vggt.v6.scheduler import initialize
    from vggt.models.aggregator import slice_expand_and_flatten
    from vggt.layers.rope import PositionGetter
    torch.set_num_threads(4)
    saved=torch.load(OLD/'inputs.pt',map_location='cpu',weights_only=True)
    manifest=json.loads((OLD/'independent/run_manifest.json').read_text())
    with safe_open(manifest['checkpoint'],framework='pt',device='cpu') as f:
        camera=f.get_tensor('aggregator.camera_token')
        register=f.get_tensor('aggregator.register_token')
    agg=SimpleNamespace(training=False,aa_order=['frame','global'],aa_block_size=1,
        camera_token=camera,register_token=register,patch_start_idx=5,patch_size=14,
        rope=True,position_getter=PositionGetter(),
        _resnet_mean=torch.tensor([.485,.456,.406]).reshape(1,1,3,1,1),
        _resnet_std=torch.tensor([.229,.224,.225]).reshape(1,1,3,1,1),
        patch_embed=lambda x:torch.zeros(x.shape[0],28*37,1024))
    rows=[]
    with torch.inference_mode():
        states,positions,count=initialize(agg,saved['images'],WINDOWS)
        for w,(lo,hi) in enumerate(WINDOWS):
            n=hi-lo
            expected=torch.cat([slice_expand_and_flatten(camera,1,n),
                                slice_expand_and_flatten(register,1,n)],dim=1)
            actual=states[w].reshape(n,count,1024)[:,:5]
            pos=PositionGetter()(n,28,37,device=torch.device('cpu'))+1
            native_pos=torch.cat([torch.zeros(n,5,2,dtype=pos.dtype),pos],dim=1).reshape(1,n*count,2)
            assert torch.equal(expected,actual) and torch.equal(native_pos,positions[w])
            rows.append(dict(window=w,first_reference=saved['frame_ids'][lo],
                camera_register_exact=True,rope_exact=True,token_hash=tensor_hash(actual),
                rope_hash=tensor_hash(positions[w]),input_slice_hash=tensor_hash(saved['images'][lo:hi])))
    write_json(out/'initialization_checks.json',dict(checks=rows,
        checkpoint_camera_sha256=tensor_hash(camera),checkpoint_register_sha256=tensor_hash(register),
        note='Real checkpoint special tokens and actual 392x518 grid. Encoder is stubbed with zeros only for CPU initialization isolation; encoder/native prediction equivalence separately verified on GPU.'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['audit','init-audit','native','native-default'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.is_dir():raise ValueError('output must be an existing unique diagnostic directory')
    if args.action=='audit':audit(args.output)
    elif args.action=='init-audit':initialization_audit(args.output)
    elif args.action.startswith('native'):native_check(args.output,args.action=='native-default')

if __name__=='__main__':main()
