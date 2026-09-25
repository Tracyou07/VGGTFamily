"""Frozen VGGT-Long adjacent point-head alignment; no GT, ICP or fallback.

The numeric algorithm is vendored byte-for-byte. Added boundary checks reject
invalid/degenerate input instead of silently filtering or recovering it.
"""
from dataclasses import dataclass,asdict
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from experiments.ours_v3.geometry import Sim3,transform_predictions,append_unique
from experiments.ours_v3.stitch import json_write

ROOT=Path(__file__).resolve().parents[2]

@lru_cache(maxsize=1)
def load_long():
    folder=ROOT/'vendor/vggtlong'
    provenance=json.loads((folder/'PROVENANCE.json').read_text())
    for name,digest in provenance['sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('Long vendor hash mismatch: '+name)
    path=folder/'loop_utils/sim3utils.py'
    spec=importlib.util.spec_from_file_location('_ours_v5_long_sim3',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

@dataclass(frozen=True)
class LongAlignmentConfig:
    delta:float=.1
    max_iters:int=5
    tol:float=1e-9
    align_method:str='numba'
    min_pairs:int=3
    degeneracy_ratio:float=1e-6

    def __post_init__(self):
        if self.align_method not in ('numba','numpy') or self.max_iters<1 or self.min_pairs<3:raise ValueError('invalid Long alignment configuration')
        if not all(np.isfinite(v) and v>0 for v in (self.delta,self.tol,self.degeneracy_ratio)):raise ValueError('invalid positive thresholds')
    def vendor_config(self):
        # The fixed upstream routine evaluates tol as a string. Never accept arbitrary code.
        return {'Model':{'align_method':self.align_method,'using_sim3':True,
                'IRLS':{'delta':self.delta,'max_iters':self.max_iters,'tol':repr(float(self.tol))}}}


def validate_prediction(pred):
    ids=list(pred['frame_ids'])
    if not ids or len(ids)!=len(set(ids)):raise ValueError('empty/duplicate frame IDs')
    points=np.asarray(pred['world_points']);conf=np.asarray(pred['world_points_conf'])
    if points.ndim!=4 or points.shape[-1]!=3 or points.shape[0]!=len(ids) or conf.shape!=points.shape[:-1]:raise ValueError('point/confidence shape mismatch')
    if not np.isfinite(points).all() or not np.isfinite(conf).all() or (conf<0).any():raise ValueError('nonfinite points or invalid confidence')
    return ids,points,conf


def _nondegenerate(points,ratio):
    x=np.asarray(points,dtype=np.float64);x=x-x.mean(0)
    eigen=np.linalg.eigvalsh(x.T@x)
    # Rank two is sufficient for proper 3-D rigid alignment; lines are not.
    if eigen[-1]<=1e-20 or eigen[-2]<=ratio**2*eigen[-1]:raise ValueError('degenerate point geometry (rank < 2)')


def align_overlap(a,b,config):
    ai,ap,ac=validate_prediction(a);bi,bp,bc=validate_prediction(b)
    lookup={frame:i for i,frame in enumerate(bi)}
    common=[frame for frame in ai if frame in lookup]
    if not common:raise ValueError('no common frame IDs')
    ia=[ai.index(f) for f in common];ib=[lookup[f] for f in common]
    ap,ac,bp,bc=ap[ia],ac[ia],bp[ib],bc[ib]
    if ap.shape!=bp.shape:raise ValueError('overlap pixel grids differ')
    threshold=float(.1*min(np.median(ac),np.median(bc)))
    mask=(ac>threshold)&(bc>threshold)
    source=bp[mask];target=ap[mask]
    if len(source)<config.min_pairs:raise ValueError('insufficient valid overlap correspondences')
    _nondegenerate(source,config.degeneracy_ratio);_nondegenerate(target,config.degeneracy_ratio)
    s,R,t=load_long().weighted_align_point_maps(ap,ac,bp,bc,None,threshold,config.vendor_config())
    if not np.isfinite(s) or s<=0 or not np.isfinite(R).all() or not np.isfinite(t).all():raise ValueError('nonfinite/nonpositive Sim3')
    if not np.allclose(R.T@R,np.eye(3),atol=5e-4,rtol=0) or not np.isclose(np.linalg.det(R),1,atol=5e-4):raise ValueError('invalid proper rotation')
    transform=Sim3(float(s),R,t)
    residual=np.linalg.norm(transform.apply(source)-target,axis=1)
    if not np.isfinite(residual).all():raise ValueError('nonfinite alignment residual')
    stats=dict(common_frame_ids=common,pairs=len(source),confidence_threshold=threshold,
               mean_residual=float(residual.mean()),rmse=float(np.sqrt(np.mean(residual**2))),max_residual=float(residual.max()),
               huber_quadratic_count=int((residual<=config.delta).sum()),huber_delta=config.delta,
               method='original Long confidence-weighted Sim3 + Huber IRLS; no RANSAC',
               residual_definition='Euclidean distance in preceding window coordinates')
    return transform,stats


class Stitcher:
    def __init__(self,directory,config=None):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.config=config or LongAlignmentConfig()
        self.previous=None;self.global_transform=Sim3(1.,np.eye(3),np.zeros(3))
        self.seen=set();self.frames=[];self.poses=[];self.intrinsics=[];self.sources=[]
    def add(self,prediction,window_id):
        edge=self.directory/f'edge_{window_id-1:04d}_{window_id:04d}.json'
        try:
            validate_prediction(prediction)
            if window_id!=len(list(self.directory.glob('window_*_transform.json'))):raise ValueError('nonsequential window ID')
            if self.previous is not None:
                adjacent,stats=align_overlap(self.previous,prediction,self.config)
                self.global_transform=self.global_transform.compose(adjacent)
                json_write(edge,dict(status='success',direction='B_local -> A_local',composition='S_B_global = S_A_global compose S_B_to_A',
                                     adjacent=adjacent.record(),global_transform=self.global_transform.record(),**stats))
            pose,depth=transform_predictions(prediction['c2w'],prediction['depth'],self.global_transform)
            if not np.isfinite(pose).all() or not np.isfinite(depth).all():raise ValueError('nonfinite transformed camera/depth')
            points=self.global_transform.apply(prediction['world_points'])
            fresh=append_unique(self.seen,list(prediction['frame_ids']))
            for i in fresh:
                self.frames.append(prediction['frame_ids'][i]);self.poses.append(pose[i]);self.intrinsics.append(prediction['intrinsics'][i]);self.sources.append(window_id)
            json_write(self.directory/f'window_{window_id:04d}_transform.json',dict(direction='window_local -> global',**self.global_transform.record(),appended_frame_ids=[prediction['frame_ids'][i] for i in fresh]))
            self.previous=prediction
            # Original pose encoding is local; keep it only in local.npz.
            transformed={k:v for k,v in prediction.items() if k!='pose_encoding'}
            transformed.update(c2w=pose,depth=depth,world_points=points)
            return fresh,transformed
        except Exception as error:
            json_write(edge,dict(status='failed',reason=str(error),traceback=traceback.format_exc()))
            raise
    def finish(self,expected_ids):
        if self.frames!=list(expected_ids):raise ValueError('final frame mapping is not exact')
        return dict(frame_ids=np.asarray(self.frames),c2w=np.asarray(self.poses),intrinsics=np.asarray(self.intrinsics),source_window=np.asarray(self.sources))
