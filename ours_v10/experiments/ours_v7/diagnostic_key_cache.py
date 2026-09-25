"""Opt-in per-window cache of the same BF16 K conversion normally repeated by SDPA autocast."""
from contextlib import contextmanager
from contextvars import ContextVar
import torch

@contextmanager
def cached_sdpa_keys():
    """Keep query tiling, K visibility/order, one softmax, projections and synchronization unchanged."""
    import vggt.v7.attention as module
    original_attend=module._attend
    original_exchange=module.exchange_attention
    current=ContextVar('v7_diagnostic_sdpa_key_cache',default=None)
    stats=dict(casts=0,hits=0,cast_bytes=0,active_window_caches=0,max_cached_bytes=0)
    def attend(attention,q,k,v):
        cache=current.get()
        if (cache is None or torch.is_grad_enabled() or not attention.fused_attn
                or not torch.is_autocast_enabled(q.device.type)):
            return original_attend(attention,q,k,v)
        dtype=torch.get_autocast_dtype(q.device.type)
        if k.dtype==dtype:return original_attend(attention,q,k,v)
        key=(id(k),dtype)
        if key not in cache:
            # torch autocast would make this identical conversion on every call.
            converted=k.to(dtype=dtype)
            cache[key]=(k,converted)
            stats['casts']+=1;stats['cast_bytes']+=converted.numel()*converted.element_size()
            stats['max_cached_bytes']=max(stats['max_cached_bytes'],sum(x[1].numel()*x[1].element_size() for x in cache.values()))
        else:stats['hits']+=1
        return original_attend(attention,q,cache[key][1],v)
    def exchange(*args,**kwargs):
        token=current.set({});stats['active_window_caches']+=1
        try:return original_exchange(*args,**kwargs)
        finally:
            current.reset(token);stats['active_window_caches']-=1
    module._attend=attend;module.exchange_attention=exchange
    try:yield stats
    finally:
        module._attend=original_attend;module.exchange_attention=original_exchange
