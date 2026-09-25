"""One shared aggregator, window heads, prediction-only sequential Sim(3)."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import platform
import resource
import subprocess
import time
import traceback

import numpy as np
import torch

from vggt.layers.overlap_windows import make_windows, slice_features
from .geometry import AlignmentConfig, fit_sim3, unproject_pixels
from .stitch import Stitcher, json_write

BASE_COMMIT='28577d4e5ab26c9ce44294f02e7da99a9e6e1ea5'
CHECKPOINT_SHA256='f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
REPO=Path(__file__).resolve().parents[2]


@dataclass
class Config:
    attention_mode: str = 'camera_global'
    window_size: int = 30
    overlap: int = 10
    camera_query_chunk_size: int = 64
    decoder_mode: str = 'windowed'
    alignment_mode: str = 'overlap_sim3'
    checkpoint: str = '/data/yjh/share/pretrained/VGGT-1B/model.safetensors'
    checkpoint_sha256: str = CHECKPOINT_SHA256
    scene_root: str = '/data/yjh/share/datasets/ScanNet/prepared_scannet50_v1/scene0150_00'
    frame_manifest: str = 'configs/scene0150_00_frames100.json'
    frames: int = 100
    device: str = 'cuda:0'
    cache_device: str = 'cpu'
    ply_pixel_stride: int = 16
    single_window_gate: bool = False
    gate_atol: float = 1e-5
    gate_rtol: float = 1e-5
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    def validate(self):
        make_windows(self.frames,self.window_size,self.overlap)
        if self.attention_mode not in ('camera_global','local_shared_ref'):
            raise ValueError('unsupported attention mode')
        if self.decoder_mode!='windowed' or self.alignment_mode!='overlap_sim3':
            raise ValueError('only windowed / overlap_sim3 supported')
        if self.camera_query_chunk_size<1 or self.ply_pixel_stride<1:
            raise ValueError('chunk and sampling stride must be positive')
        if self.cache_device not in ('cpu','gpu'): raise ValueError('invalid cache device')
        self.alignment.validate()
        if self.single_window_gate and len(make_windows(self.frames,self.window_size,self.overlap))!=1:
            raise ValueError('single-window gate requires N <= stride under the exact window formula')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()


def resolve_frames(config):
    root=Path(config.scene_root)
    manifest=Path(config.frame_manifest)
    if not manifest.is_absolute(): manifest=REPO/manifest
    record=json.loads(manifest.read_text())
    ids=record['frame_ids']
    if record.get('scene',root.name)!=root.name: raise ValueError('frame manifest scene mismatch')
    if len(set(ids))!=len(ids) or len(ids)<config.frames:
        raise ValueError('duplicate IDs or insufficient manifest frames; never repeat frames')
    ids=ids[:config.frames]
    by_id={path.stem:path for path in (root/'color').iterdir() if path.suffix.lower() in ('.jpg','.jpeg','.png')}
    if not all(frame in by_id for frame in ids): raise ValueError('manifest RGB missing')
    return ids,[by_id[frame] for frame in ids],dict(path=str(manifest),sha256=sha256(manifest))


def tensor_bytes(values):
    return sum(value.numel()*value.element_size() for value in values if value is not None)


def decode_window(model, features, all_ids, ids, images, patch_start, device):
    """Slice every cached layer by ID, preserve the original heads and precision."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sliced=slice_features(features,all_ids,ids)
    transfer=sum(v.numel()*v.element_size() for v in sliced if v is not None and v.device!=torch.device(device))
    sliced=[None if v is None else v.to(device) for v in sliced]
    indices=[all_ids.index(frame) for frame in ids]
    local_images=images[indices].to(device).unsqueeze(0)
    transfer+=local_images.numel()*local_images.element_size() if images.device!=torch.device(device) else 0
    with torch.inference_mode(),torch.autocast(torch.device(device).type,enabled=False):
        pose=model.camera_head(sliced)[-1]
        depth,confidence=model.depth_head(sliced,images=local_images,patch_start_idx=patch_start)
        ext,intr=pose_encoding_to_extri_intri(pose.float(),image_size_hw=local_images.shape[-2:])
        bottom=torch.zeros((*ext.shape[:2],1,4),device=ext.device,dtype=ext.dtype)
        bottom[...,0,3]=1
        c2w=torch.linalg.inv(torch.cat([ext,bottom],dim=-2))
    prediction=dict(frame_ids=list(ids),pose_encoding=pose[0].float().cpu().numpy(),
                    depth=depth[0].float().cpu().numpy(),confidence=confidence[0].float().cpu().numpy(),
                    c2w=c2w[0].float().cpu().numpy(),intrinsics=intr[0].float().cpu().numpy())
    transfer+=sum(value.nbytes for value in prediction.values() if isinstance(value,np.ndarray))
    return prediction,transfer


def write_cloud(path, prediction, images, selected, stride, window_id):
    """Sample consistently transformed depth/pose; invalid depth is never displayed."""
    chunks=[]; colors=[]
    for i in selected:
        depth=prediction['depth'][i,...,0]
        rows,cols=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
        rows=rows.ravel(); cols=cols.ravel()
        conf=prediction['confidence'][i].reshape(depth.shape)
        valid=np.isfinite(depth[rows,cols])&(depth[rows,cols]>0)&np.isfinite(conf[rows,cols])
        rows=rows[valid]; cols=cols[valid]
        points=unproject_pixels(depth,prediction['intrinsics'][i],prediction['c2w'][i],rows,cols)
        valid=np.isfinite(points).all(1)
        chunks.append(points[valid]); colors.append(images[i].permute(1,2,0).numpy()[rows[valid],cols[valid]])
    xyz=np.concatenate(chunks) if chunks else np.empty((0,3))
    rgb=np.clip(np.concatenate(colors)*255,0,255).astype(np.uint8) if colors else np.empty((0,3),np.uint8)
    with Path(path).open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int window_id\nend_header\n')
        for point,color in zip(xyz,rgb):
            stream.write(' '.join(map(str,(*point,*color,window_id)))+'\n')
    return xyz,rgb,np.full(len(xyz),window_id,dtype=np.int32)


def evaluate(global_result, scene_root, output):
    """Only called after finish(): GT cannot influence correspondences or stitching."""
    ids=global_result['frame_ids']; poses=global_result['c2w']
    gt=np.stack([np.loadtxt(Path(scene_root)/'pose'/f'{frame}.txt') for frame in ids])
    if not np.isfinite(gt).all(): raise ValueError('GT has nonfinite poses; no silent filtering')
    alignment=fit_sim3(poses[:,:3,3],gt[:,:3,3])
    aligned=alignment.apply(poses[:,:3,3])
    ate=float(np.sqrt(np.mean(np.sum((aligned-gt[:,:3,3])**2,axis=1))))
    np.savez_compressed(output/'evaluation.npz',gt_c2w=gt,pred_raw=poses,aligned_centers=aligned,
                        scale=alignment.scale,rotation=alignment.rotation,translation=alignment.translation)
    adjacent=[]
    for i in range(1,len(ids)):
        pred=np.linalg.inv(poses[i-1])@poses[i]; target=np.linalg.inv(gt[i-1])@gt[i]
        angle=lambda r:float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))
        adjacent.append(dict(before=str(ids[i-1]),after=str(ids[i]),
            boundary=bool(global_result['source_window'][i]!=global_result['source_window'][i-1]),
            pred_translation_raw=float(np.linalg.norm(pred[:3,3])),
            pred_translation_scaled=float(alignment.scale*np.linalg.norm(pred[:3,3])),
            gt_translation=float(np.linalg.norm(target[:3,3])),
            pred_rotation_deg=angle(pred[:3,:3]),gt_rotation_deg=angle(target[:3,:3]),
            translation_error=float(np.linalg.norm(alignment.scale*pred[:3,3]-target[:3,3])),
            rotation_error_deg=angle(target[:3,:3].T@pred[:3,:3])))
    json_write(output/'trajectory_metrics.json',dict(ate_rmse_m=ate,alignment='one global proper Sim(3)',**alignment.record()))
    json_write(output/'adjacent_pose_errors.json',adjacent)
    json_write(output/'boundary_diagnostics.json',[row for row in adjacent if row['boundary']])
    json_write(output/'within_window_diagnostics.json',[row for row in adjacent if not row['boundary']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(10,5))
    for index,points,title in ((1,poses[:,:3,3],'Raw stitched trajectory'),(2,aligned,'One global Sim(3) / GT')):
        ax=fig.add_subplot(1,2,index,projection='3d')
        ax.scatter(*points.T,c=global_result['source_window'],s=8)
        ax.plot(*points.T,alpha=.4)
        if index==2: ax.plot(*gt[:,:3,3].T,color='black',label='GT'); ax.legend()
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(output/'trajectory.png',dpi=160); plt.close(fig)


def run(config, output):
    config.validate()
    output=Path(output)
    ids,paths,frame_source=resolve_frames(config)
    windows=make_windows(len(ids),config.window_size,config.overlap)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    timings={}; begin=time.perf_counter(); transfers=0
    manifest=dict(config=asdict(config),base_commit=BASE_COMMIT,frame_ids=ids,windows=windows,
                  frame_manifest=frame_source,coordinate_convention='saved c2w, depth=z; intrinsics unchanged by Sim3',
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  working_tree=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True),
                  python=platform.python_version(),torch=torch.__version__)
    json_write(output/'run_manifest.json',manifest)
    try:
        digest=sha256(config.checkpoint)
        if digest!=config.checkpoint_sha256: raise ValueError('checkpoint SHA-256 mismatch')
        manifest['verified_checkpoint_sha256']=digest
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        device=torch.device(config.device)
        if device.type!='cuda': raise ValueError('real-image runner requires H20 CUDA; CPU unit tests are separate')
        torch.cuda.set_device(device)
        model=VGGT().eval().requires_grad_(False)
        weights=load_file(config.checkpoint,device='cpu')
        model.load_state_dict(weights,strict=True); del weights
        model.to(device)
        images=load_and_preprocess_images([str(p) for p in paths])
        manifest['preprocessing']=dict(loader='vggt.utils.load_fn default crop',shape=list(images.shape),
                                         dtype=str(images.dtype),minimum=float(images.min()),maximum=float(images.max()))
        torch.cuda.reset_peak_memory_stats(device)
        gpu_images=images.to(device); transfers+=images.numel()*images.element_size()
        torch.cuda.synchronize(device); start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            features,patch_start=model.aggregator(gpu_images.unsqueeze(0),attention_mode=config.attention_mode,
                window_size=config.window_size,camera_query_chunk_size=config.camera_query_chunk_size,windows=windows)
        torch.cuda.synchronize(device); timings['aggregator_seconds']=time.perf_counter()-start
        if config.single_window_gate:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                reference,_=model.aggregator(gpu_images.unsqueeze(0),attention_mode='full')
            reports=[]
            for index,(a,b) in enumerate(zip(features,reference)):
                if a is not None:
                    diff=(a.float()-b.float()).abs()
                    reports.append(dict(layer=index,max_abs=float(diff.max()),mean_abs=float(diff.mean())))
            gate=dict(atol=config.gate_atol,rtol=config.gate_rtol,layers=reports)
            json_write(output/'gate.json',gate)
            for a,b in zip(features,reference):
                if a is not None: torch.testing.assert_close(a,b,atol=config.gate_atol,rtol=config.gate_rtol)
            candidate,_=decode_window(model,features,ids,ids,images,patch_start,device)
            baseline,_=decode_window(model,reference,ids,ids,images,patch_start,device)
            gate['heads']={}
            for name in ('pose_encoding','depth','confidence','c2w','intrinsics'):
                difference=np.abs(candidate[name]-baseline[name])
                gate['heads'][name]=dict(max_abs=float(difference.max()),mean_abs=float(difference.mean()))
            json_write(output/'gate.json',gate)
            for name in gate['heads']:
                np.testing.assert_allclose(candidate[name],baseline[name],atol=config.gate_atol,rtol=config.gate_rtol)
            gate['passed']=True; json_write(output/'gate.json',gate)
            del reference,candidate,baseline
        del gpu_images
        manifest['retained_feature_bytes']=tensor_bytes(features)
        start=time.perf_counter()
        if config.cache_device=='cpu':
            features=[None if value is None else value.cpu() for value in features]
            transfers+=manifest['retained_feature_bytes']
        torch.cuda.synchronize(device); timings['cache_transfer_seconds']=time.perf_counter()-start
        manifest['cpu_feature_cache_bytes']=manifest['retained_feature_bytes'] if config.cache_device=='cpu' else 0
        stitch=Stitcher(output/'alignment',config.alignment)
        timings.update(decode_seconds=0.,alignment_seconds=0.,artifact_seconds=0.)
        cloud_points=[]; cloud_colors=[]; cloud_windows=[]
        for window_id,(lo,hi) in enumerate(windows):
            start=time.perf_counter()
            prediction,bytes_moved=decode_window(model,features,ids,ids[lo:hi],images,patch_start,device)
            torch.cuda.synchronize(device); timings['decode_seconds']+=time.perf_counter()-start; transfers+=bytes_moved
            folder=output/'windows'/f'{window_id:04d}'; folder.mkdir(parents=True)
            np.savez_compressed(folder/'local.npz',**prediction)
            start=time.perf_counter()
            fresh,global_pose,global_depth=stitch.add(prediction,window_id)
            timings['alignment_seconds']+=time.perf_counter()-start
            start=time.perf_counter()
            np.savez_compressed(folder/'global_new_frames.npz',
                frame_ids=np.asarray(prediction['frame_ids'])[fresh],c2w=global_pose[fresh],
                depth=global_depth[fresh],intrinsics=prediction['intrinsics'][fresh],confidence=prediction['confidence'][fresh])
            write_cloud(folder/'local.ply',prediction,images[lo:hi],range(hi-lo),config.ply_pixel_stride,window_id)
            transformed={**prediction,'c2w':global_pose,'depth':global_depth}
            xyz,rgb,wid=write_cloud(folder/'global_new_frames.ply',transformed,images[lo:hi],fresh,config.ply_pixel_stride,window_id)
            cloud_points.append(xyz); cloud_colors.append(rgb); cloud_windows.append(wid)
            timings['artifact_seconds']+=time.perf_counter()-start
        result=stitch.finish(ids)
        np.savez_compressed(output/'global_trajectory.npz',**result)
        # Preserve a standard interactive-viewer-compatible PLY without any GT transform.
        with (output/'global_pointcloud.ply').open('w') as stream:
            total=sum(len(x) for x in cloud_points)
            stream.write(f'ply\nformat ascii 1.0\nelement vertex {total}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int window_id\nend_header\n')
            for xyz,rgb,wid in zip(cloud_points,cloud_colors,cloud_windows):
                for p,c,w in zip(xyz,rgb,wid): stream.write(' '.join(map(str,(*p,*c,w)))+'\n')
        evaluate(result,config.scene_root,output)
        timings['total_seconds']=time.perf_counter()-begin
        manifest.update(timings=timings,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            explicit_transfer_payload_bytes=transfers,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            status='complete')
        json_write(output/'run_manifest.json',manifest)
        json_write(output/'COMPLETE.json',dict(status='complete'))
        return manifest
    except Exception as error:
        manifest.update(status='failed',failure=str(error),timings=timings)
        json_write(output/'run_manifest.json',manifest)
        json_write(output/'FAILED.json',dict(reason=str(error),traceback=traceback.format_exc()))
        raise
