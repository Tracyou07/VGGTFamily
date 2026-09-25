"""Locate runtime-flag numerical divergence without saving intermediate activations."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.ours_v7.diagnostics import PARENT,OLD,configure,tensor_hash,difference,write_json,csv_rows,flatten
def run(out,full_trajectory=False):
    sys.path.insert(0,str(PARENT/'vggt'))
    from vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    configure()
    manifest=json.loads((OLD/'independent/run_manifest.json').read_text())
    saved=torch.load(OLD/'inputs.pt',map_location='cpu',weights_only=True)
    model=VGGT().eval().requires_grad_(False);model.load_state_dict(load_file(manifest['checkpoint']),strict=True);model.cuda()
    if full_trajectory:
        from experiments.ours_v7.diagnostics import convert_native
        from experiments.compare_long.evaluate import evaluate_poses
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictions=model(saved['images'].cuda())
            values=convert_native(predictions,saved['images'].shape[-2:])
        np.savez(out/'full100_strict_trajectory.npz',frame_ids=saved['frame_ids'],c2w=values['c2w'])
        metrics,_=evaluate_poses(saved['frame_ids'],values['c2w'],saved['scene_root'])
        write_json(out/'full100_strict_evaluation.json',metrics)
        print('full100_strict',metrics,flush=True)
        return
    rows=[];references={};hashes={};current=''
    handles=[]
    def hook(name,value):
        ts=flatten(value)
        if not ts:return
        # One tensor at a time; bytes are hashed then discarded. No activation file.
        t=ts[0].detach().cpu()
        h=tensor_hash(t)
        row=dict(setting=current,stage=name,hash=h,shape=list(t.shape),dtype=str(t.dtype))
        if current=='strict':hashes[name]=h
        row['matches_strict_hash']=h==hashes.get(name)
        rows.append(row)
    names=['aggregator.patch_embed.patch_embed']+[f'aggregator.patch_embed.blocks.{i}' for i in range(24)]
    for name,module in model.named_modules():
        if name in names:
            handles.append(module.register_forward_hook(lambda m,a,o,n=name:hook(n,o)))
    encoder_ref=None;checks=[]
    for current,det,tf32 in [('strict',True,False),('tf32_only',True,True),('nondeterministic_only',False,False),('historical_defaults',False,True)]:
        torch.use_deterministic_algorithms(det);torch.backends.cudnn.allow_tf32=tf32
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            x=saved['images'][:60].cuda()[None]
            x=(x-model.aggregator._resnet_mean)/model.aggregator._resnet_std
            encoded=model.aggregator.patch_embed(x.reshape(60,3,392,518))['x_norm_patchtokens']
            a=encoded.float().cpu().numpy()
            if encoder_ref is None:encoder_ref=a
            checks.append(dict(setting=current,stage='image_encoder',**difference(encoder_ref,a)))
            del encoded,x
        print(current,checks[-1],flush=True)
        csv_rows(out/'runtime_flag_encoder.csv',checks);write_json(out/'runtime_flag_layer_hashes.json',rows)
    for h in handles:h.remove()
    first={}
    for setting in ['tf32_only','nondeterministic_only','historical_defaults']:
        first[setting]=next((r['stage'] for r in rows if r['setting']==setting and not r['matches_strict_hash']),None)
    write_json(out/'runtime_flag_first_divergence.json',dict(first_divergence=first,
        interpretation='BF16 is held fixed; deterministic SDPA policy and cudnn TF32 are separately controlled; no tolerance changes'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--full-trajectory',action='store_true');a=p.parse_args();run(a.output,a.full_trajectory)
