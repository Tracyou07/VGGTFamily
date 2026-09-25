"""CPU-side contracts, provenance and launch checks. No model imports at import time."""
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import numpy as np
import torch
from experiments.sevenscenes.common import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]


def fresh_directory(path):
    Path(path).mkdir(parents=True,exist_ok=False)


def prepare_inputs(scene,frame_list,output,frames):
    scene=Path(scene);output=Path(output)
    ids=json.loads(Path(frame_list).read_text())['frame_ids']
    if frames<1 or len(ids)<frames or len(set(ids))!=len(ids):raise ValueError('invalid fixed frame list; no resampling')
    if output.exists():raise FileExistsError(output)
    ids=ids[:frames];by_id={f.stem:f for f in (scene/'color').iterdir()}
    paths=[str(by_id[i]) for i in ids]
    from vggt.utils.load_fn import load_and_preprocess_images
    start=time.perf_counter();images=load_and_preprocess_images(paths);seconds=time.perf_counter()-start
    data=dict(images=images,frame_ids=ids,scene_root=str(scene),rgb_paths=paths,frame_list_sha256=sha256(frame_list),preprocessing=dict(loader='original VGGT* load_and_preprocess_images default crop',shape=list(images.shape),dtype=str(images.dtype),minimum=float(images.min()),maximum=float(images.max()),elapsed_seconds=seconds))
    # Exclusive creation: two launchers must never overwrite a prepared tensor.
    with output.open('xb') as stream:torch.save(data,stream)
    return data


def source_identity():
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    extra=[str(p.relative_to(ROOT)) for folder in ('vggt/v6','experiments/ours_v6','vendor/vggtlong') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    paths=sorted(set(tracked+extra))
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
                sha256={p:sha256(ROOT/p) for p in paths if (ROOT/p).is_file()})


def preflight(gpu,output_parent,min_disk_gib=20):
    if not str(gpu).isdigit():raise ValueError('GPU must be a physical numeric index')
    def run(args):return subprocess.check_output(args,text=True).strip()
    gpu_rows=run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
    parts=[s.strip() for s in gpu_rows.split(',')]
    if len(parts)!=5:raise ValueError('expected one physical GPU')
    uuid,name,used,free,util=parts
    jobs=run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'])
    if socket.gethostname()!='VM-0-11-ubuntu' or getpass.getuser()!='ubuntu' or 'H20' not in name:raise RuntimeError('unexpected H20 identity')
    if any(line.split(',')[0].strip()==uuid for line in jobs.splitlines()) or float(used)>512 or float(util)>0:raise RuntimeError('selected GPU is occupied; refusing to start')
    parent=Path(output_parent)
    while not parent.exists():parent=parent.parent
    disk=shutil.disk_usage(parent)
    if disk.free<min_disk_gib*2**30:raise RuntimeError('insufficient disk reserve')
    return dict(host=socket.gethostname(),user=getpass.getuser(),gpu_index=int(gpu),gpu_uuid=uuid,gpu_name=name,memory_used_mib=float(used),memory_free_mib=float(free),disk_free_bytes=disk.free,active_gpu_jobs=jobs,active_processes=run(['ps','-eo','pid,etime,comm']),python=os.sys.version)


def write_point_cloud(path,prediction,images,selected,window_id,stride=16):
    rows=[];colors=[]
    for i in selected:
        pts=prediction['world_points'][i,::stride,::stride].reshape(-1,3)
        conf=prediction['world_points_conf'][i,::stride,::stride].reshape(-1)
        color=images[i].permute(1,2,0).numpy()[::stride,::stride].reshape(-1,3)
        keep=np.isfinite(pts).all(1)&np.isfinite(conf)
        rows.append(pts[keep]);colors.append(color[keep])
    xyz=np.concatenate(rows) if rows else np.empty((0,3));rgb=np.clip(np.concatenate(colors)*255,0,255).astype(np.uint8) if colors else np.empty((0,3),np.uint8)
    with Path(path).open('x') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int window_id\nend_header\n')
        for point,color in zip(xyz,rgb):stream.write(' '.join(map(str,(*point,*color,window_id)))+'\n')
    return xyz,rgb


def data_identity(scene,frame_list):
    scene=Path(scene).resolve();ids=json.loads(Path(frame_list).read_text())['frame_ids']
    if len(ids)!=len(set(ids)) or len(ids)<1:raise ValueError('expected fixed list with >=100 unique frames')
    by_id={f.stem:f for f in (scene/'color').iterdir()}
    rows=[]
    for frame in ids:
        path=by_id[frame];stat=path.stat();rows.append((str(path),stat.st_size,stat.st_mtime_ns))
    return dict(scene_root=str(scene),frame_list_sha256=sha256(frame_list),rgb_metadata_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest())
