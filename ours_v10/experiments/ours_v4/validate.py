"""Ordered validation. A failed subprocess or comparison stops the campaign."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]

def compare(left,right,tolerance):
    report={}; passed=True
    for file in sorted(left.glob('window_*.pt')):
        a=torch.load(file,weights_only=True); b=torch.load(right/file.name,weights_only=True)
        rows={}
        for key in a:
            x=a[key].reshape(-1); y=b[key].reshape(-1)
            maximum=total=relative=0.; bad=0
            for i in range(0,x.numel(),1000000):
                u=x[i:i+1000000].float(); v=y[i:i+1000000].float()
                delta=(u-v).abs(); finite=torch.isfinite(u)&torch.isfinite(v)
                maximum=max(maximum,float(delta.max())); total+=float(delta.double().sum())
                relative=max(relative,float((delta/u.abs().clamp_min(tolerance['atol'])).max()))
                bad+=int((~finite | (delta>tolerance['atol']+tolerance['rtol']*u.abs())).sum())
            rows[key]=dict(max_abs=maximum,mean_abs=total/x.numel(),max_relative_with_atol_floor=relative,failed_elements=bad)
            # Quaternion encoding is diagnostic; matrix and geometric pose comparisons decide equivalence.
            if key!='pose_encoding': passed &= bad==0
        pa=a['c2w'].double().numpy(); pb=b['c2w'].double().numpy()
        centers=np.linalg.norm(pa[:,:3,3]-pb[:,:3,3],axis=-1)
        # Project float matrices to SO(3) before measuring tiny rotation differences.
        def rotation(r):
            u,_,v=np.linalg.svd(r); return u@v
        ra=rotation(pa[:,:3,:3]); rb=rotation(pb[:,:3,:3])
        angles=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',ra,rb)-1)/2,-1,1)))
        rows['pose_geometry']=dict(max_center_m=float(centers.max()),max_rotation_deg=float(angles.max()))
        passed &= centers.max()<=tolerance['center_m'] and angles.max()<=tolerance['rotation_deg']
        report[file.name]=rows
        del a,b
    if not report: raise ValueError('no gate captures')
    return dict(passed=bool(passed),tolerance=tolerance,windows=report)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/v4_validation.json').read_text())
    (out/'config.json').write_text(json.dumps(config,indent=2))
    try:
        from vggt.utils.load_fn import load_and_preprocess_images
        scene=Path(config['scene_root']); ids=json.loads((ROOT/'configs/scene0150_00_frames100.json').read_text())['frame_ids']
        by_id={f.stem:f for f in (scene/'color').iterdir()}
        images=load_and_preprocess_images([str(by_id[i]) for i in ids])
        torch.save(dict(images=images,frame_ids=ids,scene_root=str(scene),preprocessing=dict(loader='original default crop',shape=list(images.shape),dtype=str(images.dtype),minimum=float(images.min()),maximum=float(images.max()))),out/'inputs.pt')
        del images
        def run(name,precision,reference,capture,frames):
            command=[sys.executable,'-B','-u','-m','experiments.ours_v4.worker','--precision',precision,'--input',str(out/'inputs.pt'),'--output',str(out/name),'--frames',str(frames),'--window-batch-size','2']
            if reference: command+=['--reference']
            if capture: command+=['--capture']
            print('START',name,flush=True)
            with (out/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for precision in ('fp32','bf16'):
            run(precision+'_reference',precision,True,True,55)
            run(precision+'_repeat',precision,True,True,55)
            repeat=compare(out/(precision+'_reference'),out/(precision+'_repeat'),config['tolerances'][precision])
            (out/(precision+'_repeat_report.json')).write_text(json.dumps(repeat,indent=2))
            if not repeat['passed']: raise RuntimeError(precision+' reference repeat failed')
            run(precision+'_packed',precision,False,True,55)
            result=compare(out/(precision+'_reference'),out/(precision+'_packed'),config['tolerances'][precision])
            (out/(precision+'_equivalence_report.json')).write_text(json.dumps(result,indent=2))
            if not result['passed']: raise RuntimeError(precision+' packed equivalence failed; no 100-frame runs')
        run('reference100','bf16',True,False,100)
        run('packed100','bf16',False,False,100)
        (out/'COMPLETE.json').write_text(json.dumps(dict(status='complete')))
    except Exception as error:
        (out/'FAILED.json').write_text(json.dumps(dict(reason=str(error))))
        raise

if __name__=='__main__': main()
