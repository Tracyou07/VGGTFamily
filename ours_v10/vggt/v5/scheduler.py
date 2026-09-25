"""One-scene layer-synchronous execution, GPU-resident window instances.

No feature state is shared across overlap instances. Compute groups only affect
frame attention/encoder/head batching; camera exchange always spans the scene.
"""
import torch
from vggt.models.aggregator import slice_expand_and_flatten
from experiments.ours_v4.core import pack_groups
from .attention import global_step


from experiments.ours_v5.windows import make_windows


def _validate(a,images,windows,batch_size):
    if a.training:raise ValueError('v5 requires eval mode')
    if images.ndim!=4 or images.shape[1]!=3:raise ValueError('one scene [N,3,H,W] required')
    if a.aa_order!=['frame','global'] or a.aa_block_size!=1:raise ValueError('unsupported backbone execution order')
    if not windows or batch_size<1:raise ValueError('empty windows/invalid batch size')
    coverage=set()
    for lo,hi in windows:
        if not 0<=lo<hi<=len(images):raise ValueError('window outside input')
        coverage.update(range(lo,hi))
    if coverage!=set(range(len(images))):raise ValueError('windows must cover the scene exactly')


def initialize(a,images,windows,batch_size):
    _validate(a,images,windows,batch_size)
    device=a.camera_token.device
    states=[None]*len(windows);positions=[None]*len(windows)
    for group in pack_groups(windows,batch_size):
        batch=torch.stack([images[windows[i][0]:windows[i][1]] for i in group]).to(device)
        b,s,_,h,w=batch.shape
        norm=(batch-a._resnet_mean)/a._resnet_std
        patch=a.patch_embed(norm.reshape(b*s,3,h,w))
        if isinstance(patch,dict):patch=patch['x_norm_patchtokens']
        camera=slice_expand_and_flatten(a.camera_token,b,s)
        register=slice_expand_and_flatten(a.register_token,b,s)
        tokens=torch.cat([camera,register,patch],dim=1)
        p,c=tokens.shape[-2:]
        pos=None
        if a.rope is not None:
            spatial=a.position_getter(b*s,h//a.patch_size,w//a.patch_size,device=device)+1
            special=torch.zeros(b*s,a.patch_start_idx,2,device=device,dtype=spatial.dtype)
            pos=torch.cat([special,spatial],dim=1).reshape(b,s*p,2)
        tokens=tokens.reshape(b,s*p,c)
        for j,i in enumerate(group):
            states[i]=tokens[j:j+1].clone()
            positions[i]=None if pos is None else pos[j:j+1].clone()
    return states,positions,p


def aggregate_windows(a,images,windows,mode,batch_size=2,camera_query_chunk_size=64,reverse=False):
    if mode not in ('independent','camera_exchange'):raise ValueError('unknown communication mode')
    states,positions,p=initialize(a,images,windows,batch_size)
    c=states[0].shape[-1];features=[[None]*a.depth for _ in windows]
    groups=pack_groups(windows,batch_size)
    if reverse:groups=groups[::-1]
    for layer in range(a.depth):
        # Finish this layer's frame block for EVERY window before banking cameras.
        frame_cache={}
        for group in groups:
            s=windows[group[0]][1]-windows[group[0]][0]
            x=torch.cat([states[i] for i in group],dim=0).reshape(len(group)*s,p,c)
            pos=None if positions[group[0]] is None else torch.cat([positions[i] for i in group]).reshape(len(group)*s,p,2)
            y=a.frame_blocks[layer](x,pos=pos).reshape(len(group),s*p,c)
            for j,i in enumerate(group):
                states[i]=y[j:j+1].clone()
                if layer in a.cached_layer_indices:frame_cache[i]=states[i].reshape(1,s,p,c)
        order=list(range(len(states)))
        if reverse:order.reverse()
        states=global_step(a.global_blocks[layer],states,positions,p,mode,camera_query_chunk_size,order)
        if layer in a.cached_layer_indices:
            for i,(lo,hi) in enumerate(windows):
                features[i][layer]=torch.cat([frame_cache[i],states[i].reshape(1,hi-lo,p,c)],dim=-1)
    return features,a.patch_start_idx
