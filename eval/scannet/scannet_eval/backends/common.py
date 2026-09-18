"""Validated output conversion; no model imports or GPU work at import time."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

@dataclass
class Prediction:
    points: np.ndarray
    poses_c2w: np.ndarray
    frame_ids: tuple[int, ...]
    inference_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    metadata: dict

def load_state(path):
    import torch
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(str(path), map_location='cpu', weights_only=True)
    for key in ('state_dict', 'model'):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('checkpoint must contain a nonempty tensor state dictionary')
    return state

def checked_load(model, state, allowed_unused=()):
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(k for k in set(state) - set(expected) if not k.startswith(allowed_unused))
    mismatched = sorted(k for k in set(expected) & set(state) if expected[k].shape != state[k].shape)
    if missing or unexpected or mismatched:
        raise ValueError(f'incompatible checkpoint: missing={missing[:12]}, unexpected={unexpected[:12]}, shape={mismatched[:12]}')
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    return {'loaded_keys': len(expected), 'unused_keys': sorted(set(state) - set(expected))}

def validate_scene(scene):
    ids, paths = tuple(scene.frame_ids), tuple(scene.image_paths)
    if not ids or len(ids) != len(paths) or len(set(ids)) != len(ids):
        raise ValueError('requested frames must be nonempty, unique and match image count')
    if any(not isinstance(i, (int, np.integer)) for i in ids):
        raise ValueError('original frame IDs must be integers')
    if any(not Path(p).is_file() for p in paths):
        raise ValueError('requested frame image is missing')
    return ids, [str(Path(p).resolve()) for p in paths]

def validate_poses(poses, count):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        raise ValueError('pose count/shape/finite validation failed')
    R = poses[:, :3, :3]
    if not np.allclose(poses[:,3], [0,0,0,1], atol=1e-5) or not np.allclose(R @ R.transpose(0,2,1), np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(R), 1, atol=2e-3):
        raise ValueError('predicted camera poses must be proper rigid c2w transforms')
    return poses

def w2c_to_c2w(extrinsics):
    extrinsics = np.asarray(extrinsics)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3,4), (4,4)):
        raise ValueError('w2c must have shape [N,3,4] or [N,4,4]')
    matrices = np.broadcast_to(np.eye(4), (len(extrinsics),4,4)).copy()
    matrices[:,:extrinsics.shape[1]] = extrinsics
    validate_poses(matrices, len(matrices))
    return np.linalg.inv(matrices)

def finish_prediction(points, poses, ids, seconds, allocated, reserved, metadata, max_points=0):
    poses = validate_poses(poses, len(ids))
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3: raise ValueError('points must be [M,3]')
    valid = np.isfinite(points).all(axis=1)
    meta = dict(metadata, invalid_points_removed=int((~valid).sum()), max_points=int(max_points))
    points = points[valid]
    if not len(points): raise ValueError('no finite predicted points')
    meta['points_before_cap'] = len(points)
    if max_points < 0: raise ValueError('max_points must be nonnegative')
    if max_points and len(points) > max_points:
        points = points[np.random.RandomState(33).choice(len(points), max_points, replace=False)]
    if not np.isfinite(seconds) or seconds < 0: raise ValueError('invalid inference time')
    return Prediction(points, poses, tuple(ids), float(seconds), int(allocated), int(reserved), meta)

def transform_projective(points, homography, epsilon=1e-9):
    points = np.asarray(points).reshape(-1,3)
    h = np.asarray(homography)
    if h.shape != (4,4) or not np.isfinite(h).all(): raise ValueError('invalid graph homography')
    result = np.column_stack((points, np.ones(len(points)))) @ h.T
    valid = np.isfinite(result).all(axis=1) & (np.abs(result[:,3]) > epsilon)
    output = np.full((len(points),3), np.nan)
    output[valid] = result[valid,:3] / result[valid,3,None]
    return output, valid

def extract_long_chunks(chunks, ranges, transforms, count, coefficient=.75):
    if len(chunks) != len(ranges) or len(transforms) != len(chunks)-1: raise ValueError('native chunk transform count mismatch')
    owners = np.full(count, -1, dtype=int)
    for k, (start,end) in enumerate(ranges):
        if not 0 <= start < end <= count: raise ValueError('invalid native chunk range')
        owners[start:end] = k
    if np.any(owners < 0): raise ValueError('native Long omitted requested frames')
    cameras = np.empty((count,4,4)); output=[]; thresholds=[]
    for k, (chunk,(start,end)) in enumerate(zip(chunks,ranges)):
        points=np.asarray(chunk['world_points']); conf=np.asarray(chunk['world_points_conf']); poses=np.asarray(chunk['extrinsic'])
        if len(points)!=end-start or len(poses)!=end-start or points.shape[:-1]!=conf.shape: raise ValueError('native chunk frame/point/confidence mismatch')
        s,R,t = (1.,np.eye(3),np.zeros(3)) if k==0 else transforms[k-1]
        s=float(s); R=np.asarray(R); t=np.asarray(t).reshape(3)
        if not np.isfinite(s) or s<=0 or not np.isfinite(R).all() or not np.isfinite(t).all(): raise ValueError('invalid native Sim3')
        threshold=float(np.mean(conf)*coefficient); thresholds.append(threshold)
        if not np.isfinite(threshold): raise ValueError('nonfinite native confidence threshold')
        for j,index in enumerate(range(start,end)):
            if owners[index]!=k: continue
            frame=points[j].reshape(-1,3); mask=np.isfinite(conf[j].reshape(-1)) & (conf[j].reshape(-1)>threshold)
            output.append(s*(frame[mask] @ R.T)+t)
            cameras[index]=poses[j]; cameras[index,:3,:3]=R@poses[j,:3,:3]; cameras[index,:3,3]=s*(R@poses[j,:3,3])+t
    return np.concatenate(output), cameras, {'frame_owner':owners.tolist(),'chunks':len(chunks),'confidence_rule':'conf > full_chunk_mean * coefficient','conf_threshold_coef':coefficient,'confidence_thresholds':thresholds}

def extract_slam_submaps(submaps, graph, path_ids, ids):
    owned={}; output=[]; invalid=0; regular=0
    for submap in submaps:
        if submap.get_lc_status(): continue
        regular+=1; poses=submap.get_all_poses_world(graph)
        if len(poses)!=len(submap.img_names) or len(submap.pointclouds)!=len(poses): raise ValueError('SLAM submap frame count mismatch')
        for j,name in enumerate(submap.img_names):
            if str(name) not in path_ids: raise ValueError('SLAM emitted unrequested image')
            frame_id=path_ids[str(name)]
            if frame_id in owned: continue
            points,valid=transform_projective(submap.pointclouds[j],graph.get_homography(submap.get_id()+j))
            conf=np.asarray(submap.conf_masks[j]).reshape(-1)
            if len(conf)!=len(points): raise ValueError('SLAM confidence shape mismatch')
            mask=np.isfinite(conf)&(conf>submap.conf_threshold)
            invalid+=int((mask&~valid).sum()); output.append(points[mask&valid]); owned[frame_id]=(poses[j],submap.get_id())
    if set(owned)!=set(ids): raise ValueError('SLAM frame coverage differs from request')
    return np.concatenate(output), np.stack([owned[i][0] for i in ids]), {'frame_owner':[owned[i][1] for i in ids],'submaps':regular,'projective_invalid_removed':invalid,'confidence_rule':'native submap depth_conf percentile + 1e-6'}

def stage_images(ids, paths, work_dir):
    directory=Path(work_dir)/'input_frames'; directory.mkdir(parents=True,exist_ok=False)
    staged=[]
    for index,path in enumerate(paths):
        suffix=Path(path).suffix.lower()
        if suffix not in ('.jpg','.jpeg','.png'): raise ValueError('native backend requires JPEG/PNG input')
        if suffix=='.jpeg': suffix='.jpg'
        destination=directory/f'{index:08d}{suffix}'; destination.symlink_to(path); staged.append(str(destination))
    (directory/'frame_ids.json').write_text(json.dumps({'frame_ids':list(ids),'source_paths':paths},indent=2))
    return directory,staged
