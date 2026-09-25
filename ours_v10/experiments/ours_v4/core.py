"""Batch scheduling only. No attention changes or feature caches."""
from collections import defaultdict
import torch


def pack_groups(windows, batch_size):
    if batch_size<1: raise ValueError('window_batch_size must be positive')
    groups=defaultdict(list)
    for index,(lo,hi) in enumerate(windows):
        if not 0<=lo<hi: raise ValueError('invalid interval')
        groups[hi-lo].append(index)
    return [indices[i:i+batch_size] for indices in groups.values() for i in range(0,len(indices),batch_size)]


def instance_inputs(images, windows):
    return [images[lo:hi].clone() for lo,hi in windows]


def infer_batches(model, instances, windows, batch_size):
    result={}
    for group in pack_groups(windows,batch_size):
        # stack creates separate batch storage, never flatten into a scene.
        inputs=torch.stack([instances[i] for i in group])
        out=model(inputs)
        for offset,index in enumerate(group):
            result[index]={key:value[offset].clone() for key,value in out.items() if isinstance(value,torch.Tensor)}
    return [result[i] for i in range(len(windows))]
