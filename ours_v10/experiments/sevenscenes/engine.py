"""Frozen independent-window inference and the unchanged v4 stitching algorithm."""
import importlib.util
from pathlib import Path
import time
import numpy as np
import torch
from experiments.ours_v4.core import pack_groups
from experiments.ours_v3.geometry import AlignmentConfig
from vggt.layers.overlap_windows import make_windows
from .common import point_maps


def infer_windows(model,images,ids,batch_size):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    windows=make_windows(len(ids),30,10); groups=pack_groups(windows,batch_size)
    predictions={}; timing=dict(vggt_forward=0.,packing_transfer=0.,prediction_conversion=0.)
    for group in groups:
        torch.cuda.synchronize(); start=time.perf_counter()
        batch=torch.stack([images[windows[i][0]:windows[i][1]].clone() for i in group]).cuda()
        torch.cuda.synchronize(); timing['packing_transfer']+=time.perf_counter()-start
        start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(batch)
        torch.cuda.synchronize(); timing['vggt_forward']+=time.perf_counter()-start
        start=time.perf_counter()
        with torch.inference_mode():
            ext,intr=pose_encoding_to_extri_intri(output['pose_enc'].float(),image_size_hw=batch.shape[-2:])
            bottom=torch.zeros((*ext.shape[:2],1,4),device=ext.device); bottom[...,0,3]=1
            c2w=torch.linalg.inv(torch.cat((ext,bottom),dim=-2))
        for offset,index in enumerate(group):
            lo,hi=windows[index]
            predictions[index]=dict(frame_ids=ids[lo:hi],c2w=c2w[offset].float().cpu().numpy().copy(),
                intrinsics=intr[offset].float().cpu().numpy().copy(),depth=output['depth'][offset].float().cpu().numpy().copy(),
                confidence=output['depth_conf'][offset].float().cpu().numpy().copy())
        del output,batch,ext,intr,c2w
        torch.cuda.synchronize(); timing['prediction_conversion']+=time.perf_counter()-start
    return [predictions[i] for i in range(len(windows))],windows,groups,timing


class DeferredDiagnostics:
    """Private module copy changes only diagnostic writers, never geometry/model.

    Original Stitcher is executed verbatim. Queue exports until the compute timer
    stops, so filesystem compression/write latency is not reconstruction time.
    """
    def __init__(self): self.pending=[]
    def json(self,path,value): self.pending.append(('json',path,value))
    def savez_compressed(self,path,**values): self.pending.append(('npz',path,values))
    def __getattr__(self,name): return getattr(np,name)
    def flush(self):
        from .common import write_json
        for kind,path,value in self.pending:
            if kind=='json': write_json(path,value)
            else: np.savez_compressed(path,**value)
        self.pending.clear()


def stitch_predictions(predictions,ids,folder,config=None):
    import experiments.ours_v3.stitch as original
    spec=importlib.util.spec_from_file_location('experiments.ours_v3._sevenscenes_stitch',original.__file__)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    diagnostics=DeferredDiagnostics(); module.np=diagnostics; module.json_write=diagnostics.json
    stitch=module.Stitcher(Path(folder)/'alignment',config or AlignmentConfig())
    depths=[]; confidence=[]
    start=time.perf_counter()
    try:
        for index,prediction in enumerate(predictions):
            fresh,poses,depth=stitch.add(prediction,index)
            depths.append(depth[fresh]); confidence.append(prediction['confidence'][fresh])
        result=stitch.finish(ids)
        result['depth']=np.concatenate(depths); result['confidence']=np.concatenate(confidence)
        seconds=time.perf_counter()-start
    except Exception:
        diagnostics.flush()  # Preserve failed edge and preceding transforms.
        raise
    export_start=time.perf_counter(); diagnostics.flush(); export_seconds=time.perf_counter()-export_start
    return result,seconds,export_seconds


def compare_predictions(a,b,tolerance):
    rows={}; passed=True
    if len(a)!=len(b): raise ValueError('window count mismatch')
    for index,(left,right) in enumerate(zip(a,b)):
        if list(left['frame_ids'])!=list(right['frame_ids']): raise ValueError('frame mapping mismatch')
        fields={}
        for key in ('c2w','intrinsics','depth','confidence'):
            if key not in left and key not in right: continue
            x=np.asarray(left[key],np.float64); y=np.asarray(right[key],np.float64)
            if x.shape!=y.shape: raise ValueError('output shape mismatch')
            delta=np.abs(x-y)
            bad=(~np.isfinite(x))|(~np.isfinite(y))|(delta>tolerance['atol']+tolerance['rtol']*np.abs(x))
            fields[key]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                max_relative_with_atol_floor=float((delta/np.maximum(np.abs(x),tolerance['atol'])).max()),failed_elements=int(bad.sum()))
            passed &= not bad.any()
        x=np.asarray(left['c2w'],np.float64); y=np.asarray(right['c2w'],np.float64)
        distance=np.linalg.norm(x[:,:3,3]-y[:,:3,3],axis=-1)
        u,_,v=np.linalg.svd(x[:,:3,:3]); ra=u@v
        u,_,v=np.linalg.svd(y[:,:3,:3]); rb=u@v
        angle=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',ra,rb)-1)/2,-1,1)))
        fields['pose_geometry']=dict(max_center=float(distance.max()),max_rotation_deg=float(angle.max()))
        passed &= distance.max()<=tolerance['center_m'] and angle.max()<=tolerance['rotation_deg']
        rows[str(index)]=fields
    return dict(passed=bool(passed),windows=rows,tolerance=tolerance)
