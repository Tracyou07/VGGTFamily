"""Frozen-window alignment replay using actual native Long numerical routines."""
import ast
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from experiments.ours_v7.diagnostics import (ROOT,PARENT,OLD,LONG,WINDOWS,load_ours,
    load_native_module,difference,csv_rows,write_json,tensor_hash,assemble)
from experiments.ours_v3.geometry import Sim3

def native_assembly(predictions,transforms,ownership,windows=None):
    """Execute actual Long save_camera_poses AST up to exports, without loading GPU/retrieval."""
    source=PARENT/'vggtlong/vggt_long.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VGGT_Long')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='save_camera_poses'))
    cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='poses_path' for t in n.targets))
    fn.body=fn.body[:cut]+[ast.Return(value=ast.Name(id='all_poses',ctx=ast.Load()))]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    scope={'np':np}
    exec(compile(module,str(source), 'exec'),scope)
    if windows is None:windows=WINDOWS
    camera=[];intrinsics=[];last_end=0
    for p,(lo,hi) in zip(predictions,windows):
        start=lo if ownership=='last' else max(lo,last_end)
        camera.append(((start,hi),p['c2w'][start-lo:]))
        intrinsics.append(((start,hi),None))
        last_end=hi
    dummy=SimpleNamespace(img_list=list(range(windows[-1][1])),all_camera_poses=camera,
                          all_camera_intrinsics=intrinsics,
                          sim3_list=[(t.scale,t.rotation,t.translation) for t in transforms[1:]])
    return np.stack(scope['save_camera_poses'](dummy))

def run(out):
    from vggt.v5.alignment import align_overlap,LongAlignmentConfig
    from experiments.compare_long.evaluate import evaluate_poses
    import sys
    sys.path.insert(0,str(PARENT/'vggtlong'))
    native=load_native_module('_diagnostic_native_long',PARENT/'vggtlong/loop_utils/sim3utils.py')
    cfg=LongAlignmentConfig()
    preds=[load_ours(w) for w in range(3)]
    for p in preds:
        for v in p.values():
            if isinstance(v,np.ndarray):v.flags.writeable=False
    identity=Sim3(1.,np.eye(3),np.zeros(3));ours=[identity];native_edges=[]
    rows=[];timing=[];edge_json=[]
    for w in (1,2):
        a,b=preds[w-1:w+1]
        t0=time.perf_counter();T,stats=align_overlap(a,b,cfg);ours.append(ours[-1].compose(T))
        timing.append(dict(run='alignment_replay',stage='ours_stitch_edge',edge=w-1,seconds=time.perf_counter()-t0,cold=w==1))
        ap=a['world_points'][-30:];bp=b['world_points'][:30]
        ac=a['world_points_conf'][-30:];bc=b['world_points_conf'][:30]
        threshold=min(np.median(ac),np.median(bc))*.1
        mask=(ac>threshold)&(bc>threshold);weights=np.sqrt(ac[mask]*bc[mask])
        t0=time.perf_counter()
        s,R,t=native.weighted_align_point_maps(ap,ac,bp,bc,None,threshold,cfg.vendor_config())
        timing.append(dict(run='alignment_replay',stage='long_stitch_edge',edge=w-1,seconds=time.perf_counter()-t0,cold=w==1))
        native_edges.append((s,R,t))
        rows.append(dict(edge=f'{w-1}->{w}',direction='B_local -> A_local',
                         overlap_ids_equal=list(a['frame_ids'][-30:])==list(b['frame_ids'][:30]),
                         overlap_start=str(b['frame_ids'][0]),overlap_end=str(b['frame_ids'][29]),
                         pixel_correspondence='all row-major pixels at same frame ID',
                         pairs=int(mask.sum()),finite=bool(np.isfinite(ap).all() and np.isfinite(bp).all()),
                         mask_sha256=tensor_hash(mask),weight_sha256=tensor_hash(weights),
                         confidence_threshold=float(threshold),threshold_equal=float(threshold)==stats['confidence_threshold'],
                         weight_rule='sqrt(confA*confB), normalized then Huber IRLS',
                         delta=cfg.delta,max_iters=cfg.max_iters,tol=cfg.tol,
                         scale_abs=abs(T.scale-s),rotation_max_abs=float(np.max(np.abs(T.rotation-R))),
                         translation_max_abs=float(np.max(np.abs(T.translation-t)))))
        edge_json.append(dict(edge=w-1,ours=T.record(),long=Sim3(s,R,t).record(),stats=stats))
    longs=[identity]+[Sim3(*t) for t in native.accumulate_sim3_transforms(native_edges)]
    for w in (1,2):
        rows[w-1].update(cumulative_scale_abs=abs(ours[w].scale-longs[w].scale),
                        cumulative_rotation_max_abs=float(np.max(np.abs(ours[w].rotation-longs[w].rotation))),
                        cumulative_translation_max_abs=float(np.max(np.abs(ours[w].translation-longs[w].translation))))
    csv_rows(out/'alignment_comparison.csv',rows)
    write_json(out/'alignment_details.json',edge_json)
    own=[];saved={}
    scene=json.loads((OLD/'config.json').read_text())['base']['scene_root']
    ids=[f'{i:06d}' for i in range(100)]
    for rule in ('first','last'):
        op,owners=assemble(preds,ours,rule,'ours')
        lp=native_assembly(preds,longs,rule)
        saved[rule]=op
        for path,pose in [('ours',op),('long',lp)]:
            metrics,_=evaluate_poses(ids,pose,scene)
            own.append(dict(ownership=rule,path=path,**{k:v for k,v in metrics.items() if not isinstance(v,(dict,list))},
                            **difference(op,pose)))
            np.savez(out/f'ownership_{path}_{rule}.npz',frame_ids=ids,c2w=pose,source_window=owners)
    csv_rows(out/'ownership_comparison.csv',own)
    csv_rows(out/'alignment_timing.csv',timing)
    original=np.load(OLD/'independent/global_trajectory.npz')
    write_json(out/'alignment_summary.json',dict(ours_first_vs_historical=difference(original['c2w'],saved['first']),
        first_vs_last=difference(saved['first'],saved['last']),
        native_assembly_source=str(PARENT/'vggtlong/vggt_long.py'),
        native_assembly_method='unaltered save_camera_poses computation extracted by AST; export removed; first ownership restricts ranges only',
        all_edges_exact=all(r['scale_abs']==r['rotation_max_abs']==r['translation_max_abs']==0 for r in rows)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.output)
