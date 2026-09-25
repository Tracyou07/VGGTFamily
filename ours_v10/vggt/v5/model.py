"""Frozen VGGT* wrapper; original model and prediction heads remain untouched."""
import time
import torch
from torch import nn
from experiments.ours_v4.core import pack_groups
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from .scheduler import aggregate_windows,make_windows


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


class WindowReconstructor(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model.eval().requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self,images,frame_ids,mode='camera_exchange',window_size=60,overlap=30,batch_size=2,camera_query_chunk_size=64):
        if self.training or self.model.training:raise ValueError('v5 requires eval mode')
        if images.ndim!=4 or len(frame_ids)!=len(images) or len(set(frame_ids))!=len(frame_ids):raise ValueError('one scene with unique frame IDs is required')
        if any(p.requires_grad for p in self.model.parameters()):raise ValueError('all parameters must be frozen')
        if any(getattr(self.model,name,None) is None for name in ('camera_head','depth_head','point_head')):raise ValueError('camera, depth and point heads required')
        windows=make_windows(len(images),window_size,overlap);groups=pack_groups(windows,batch_size)
        device=next(self.model.parameters()).device
        synchronize(device);start=time.perf_counter()
        features,patch_start=aggregate_windows(self.model.aggregator,images,windows,mode,batch_size,camera_query_chunk_size)
        synchronize(device);backbone=time.perf_counter()-start
        feature_bytes=sum(x.numel()*x.element_size() for f in features for x in f if x is not None)
        predictions=[None]*len(windows);head_seconds=0.;transfer_seconds=0.
        for group in groups:
            lo,hi=windows[group[0]];length=hi-lo
            synchronize(device);start=time.perf_counter()
            inputs=torch.stack([images[windows[i][0]:windows[i][1]] for i in group]).to(device)
            caches=[None if features[group[0]][layer] is None else torch.cat([features[i][layer] for i in group],dim=0)
                    for layer in range(len(features[0]))]
            with torch.autocast(device_type=device.type,enabled=False):
                poses=self.model.camera_head(caches)[-1]
                depth,depth_conf=self.model.depth_head(caches,images=inputs,patch_start_idx=patch_start)
                points,point_conf=self.model.point_head(caches,images=inputs,patch_start_idx=patch_start)
                ext,intr=pose_encoding_to_extri_intri(poses.float(),image_size_hw=inputs.shape[-2:])
                bottom=torch.zeros((*ext.shape[:2],1,4),device=device,dtype=ext.dtype);bottom[...,0,3]=1
                c2w=torch.linalg.inv(torch.cat([ext,bottom],dim=-2))
            synchronize(device);head_seconds+=time.perf_counter()-start
            start=time.perf_counter()
            values=dict(pose_encoding=poses,c2w=c2w,intrinsics=intr,depth=depth,depth_conf=depth_conf,confidence=depth_conf,world_points=points,world_points_conf=point_conf)
            for j,i in enumerate(group):
                pred={name:value[j].detach().float().cpu().clone() for name,value in values.items()}
                if not all(torch.isfinite(v).all() for v in pred.values()):raise ValueError(f'nonfinite prediction at window {i}')
                if (pred['intrinsics'][:,[0,1],[0,1]]<=0).any():raise ValueError('nonpositive focal length')
                pred['frame_ids']=list(frame_ids[windows[i][0]:windows[i][1]]);predictions[i]=pred
                features[i]=[None]*len(features[i])
            synchronize(device);transfer_seconds+=time.perf_counter()-start
            del caches,inputs,values,poses,depth,depth_conf,points,point_conf,c2w,ext,intr,bottom
        return dict(predictions=predictions,windows=windows,groups=groups,
                    timing=dict(backbone_seconds=backbone,head_seconds=head_seconds,output_transfer_seconds=transfer_seconds),
                    retained_head_features_bytes=feature_bytes,state_device=str(device),cpu_offload=False)
