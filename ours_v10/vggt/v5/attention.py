"""Exact allowed-edge attention; no full-sequence score matrix or mask.

Only original modules/parameters are used. Window instances are independent;
a scene is the entire list passed to global_step, never a compute batch.
"""
import torch
from torch.nn import functional as F


def project_qkv(attention, x, pos):
    b,n,c=x.shape
    q,k,v=attention.qkv(x).reshape(b,n,3,attention.num_heads,attention.head_dim).permute(2,0,3,1,4).unbind(0)
    q,k=attention.q_norm(q),attention.k_norm(k)
    if attention.rope is not None:
        q=attention.rope(q,pos); k=attention.rope(k,pos)
    return q,k,v


def camera_bank(attention, normalized_old_state, pos, tokens_per_frame):
    # QKV/normalization and spatial RoPE have no token-to-token dependencies.
    # Projecting camera rows alone avoids retaining every window's dense QKV.
    _,k,v=project_qkv(attention,normalized_old_state[:,::tokens_per_frame],
                      None if pos is None else pos[:,::tokens_per_frame])
    return k,v


def _attend(attention,q,k,v):
    if attention.fused_attn:
        return F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,scale=attention.scale)
    # Optional original non-fused backend, still only the allowed submatrix.
    return ((q*attention.scale)@k.transpose(-2,-1)).softmax(-1)@v


def exchange_attention(attention,x,pos,tokens_per_frame,bank,window_index,camera_query_chunk_size=64):
    if attention.training: raise ValueError('v5 attention is inference-only')
    if x.shape[0]!=1 or tokens_per_frame<1 or x.shape[1]%tokens_per_frame:
        raise ValueError('one window instance with complete frame tokens is required')
    if camera_query_chunk_size<1 or not 0<=window_index<len(bank): raise ValueError('invalid camera chunk/index')
    # With no remote window every edge is allowed. Preserve the original SDPA
    # call shape: splitting queries changes BF16 rounding even for identical K/V.
    if len(bank)==1:
        return attention(x,pos=pos)
    q,k,v=project_qkv(attention,x,pos)
    camera=torch.arange(0,x.shape[1],tokens_per_frame,device=x.device)
    other=torch.arange(x.shape[1],device=x.device)
    other=other[other%tokens_per_frame!=0]
    remote=[item for i,item in enumerate(bank) if i!=window_index]
    ck=torch.cat([k]+[item[0] for item in remote],dim=2) if remote else k
    cv=torch.cat([v]+[item[1] for item in remote],dim=2) if remote else v
    output=None
    for start in range(0,len(camera),camera_query_chunk_size):
        indices=camera[start:start+camera_query_chunk_size]
        y=_attend(attention,q[:,:,indices],ck,cv)
        if output is None: output=torch.empty(q.shape,device=q.device,dtype=y.dtype)
        output[:,:,indices]=y
    if len(other):
        output[:,:,other]=_attend(attention,q[:,:,other],k,v)
    output=output.transpose(1,2).reshape_as(x)
    return attention.proj_drop(attention.proj(output))


def global_step(block,states,positions,tokens_per_frame,mode,camera_query_chunk_size=64,order=None):
    if mode not in ('independent','camera_exchange'): raise ValueError('unknown communication mode')
    if block.training: raise ValueError('v5 is inference-only')
    if len(states)!=len(positions) or not states: raise ValueError('empty or mismatched scene')
    indices=list(range(len(states))) if order is None else list(order)
    if sorted(indices)!=list(range(len(states))): raise ValueError('order must be a permutation')
    # Capture every remote camera before writing any window's new state.
    bank=None
    if mode=='camera_exchange' and len(states)>1:
        bank=[camera_bank(block.attn,block.norm1(x[:,::tokens_per_frame]),
                          None if p is None else p[:,::tokens_per_frame],1)
              for x,p in zip(states,positions)]
    result=[None]*len(states)
    for i in indices:
        x=states[i]
        if mode=='independent' or len(states)==1:
            result[i]=block(x,pos=positions[i])
            continue
        y=exchange_attention(block.attn,block.norm1(x),positions[i],tokens_per_frame,bank,i,camera_query_chunk_size)
        x=x+block.ls1(y)
        result[i]=x+block.ls2(block.mlp(block.norm2(x)))
    return result
