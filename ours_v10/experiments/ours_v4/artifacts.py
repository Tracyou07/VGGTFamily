from pathlib import Path
import numpy as np
from experiments.ours_v3.geometry import fit_sim3,unproject_pixels
from experiments.ours_v3.stitch import json_write

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
