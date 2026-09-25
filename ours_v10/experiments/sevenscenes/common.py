"""CPU-only contracts for mapping, point maps and checked sequence resume."""
import hashlib
import json
import re
import threading
from pathlib import Path
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def frames_for_sequence(root,sequence,kf):
    if kf not in (3,10): raise ValueError('kf must be 3 or 10')
    if not re.fullmatch(r'[a-z]+/seq-\d{2}',sequence): raise ValueError('invalid sequence')
    folder=Path(root)/sequence
    paths=sorted(folder.glob('frame-*.color.png'))
    expected=[folder/f'frame-{i:06}.color.png' for i in range(len(paths))]
    if not paths or paths!=expected: raise ValueError('missing or noncontiguous RGB frame IDs')
    ids=[f'{i:06}' for i in range(0,len(paths),kf)]
    return ids,paths[::kf]


def point_maps(depth,intrinsics,c2w):
    depth=np.asarray(depth)
    if depth.ndim==4: depth=depth[...,0]
    n,h,w=depth.shape
    rows,cols=np.mgrid[:h,:w]
    pixels=np.stack((cols,rows,np.ones_like(cols)),axis=-1)
    result=np.empty((n,h,w,3),np.float32)
    for i in range(n):
        rays=pixels@np.linalg.inv(intrinsics[i]).T
        result[i]=(rays*depth[i,...,None])@c2w[i,:3,:3].T+c2w[i,:3,3]
    if not np.isfinite(result).all(): raise ValueError('nonfinite predicted point map')
    return result


def check_row(row):
    for key in ('acc','comp','nc1','nc2','mean_nc'):
        if not np.isfinite(row[key]): raise ValueError('nonfinite metric: '+key)
    if row['acc']<0 or row['comp']<0: raise ValueError('negative distance')
    if not all(0<=row[k]<=1.00001 for k in ('nc1','nc2')): raise ValueError('invalid normal consistency')
    if not np.isclose(row['mean_nc'],(row['nc1']+row['nc2'])/2): raise ValueError('incorrect Mean NC')


def summarize(rows,expected):
    ids=[row['scene_id'] for row in rows]
    if len(set(ids))!=len(ids) or not set(ids)<=set(expected): raise ValueError('duplicate/unexpected sequences')
    for row in rows: check_row(row)
    result={key:float(np.mean([r[key] for r in rows])) if rows else None for key in ('acc','comp','nc1','nc2','mean_nc')}
    result.update(valid_sequences=len(rows),expected_sequences=len(expected),complete=len(expected)==18 and set(ids)==set(expected),
                  missing_sequences=[s for s in expected if s not in ids])
    return result


def seal_attempt(folder,fingerprint,sequence,ids):
    folder=Path(folder)
    names=['metrics.json','diagnostics.json','trajectory.npz']
    names += [str(p.relative_to(folder)) for p in sorted((folder/'alignment').glob('*')) if p.is_file()]
    names += [str(p.relative_to(folder)) for p in sorted((folder/'windows').rglob('*.npz'))]
    write_json(folder/'COMPLETE.json',dict(fingerprint=fingerprint,scene_id=sequence,frame_ids=ids,
        artifacts={name:sha256(folder/name) for name in names}))


def valid_attempt(folder,fingerprint,sequence,ids):
    folder=Path(folder)
    try:
        if (folder/'FAILED.json').exists(): return None
        seal=json.loads((folder/'COMPLETE.json').read_text())
        if (seal['fingerprint'],seal['scene_id'],seal['frame_ids'])!=(fingerprint,sequence,ids): return None
        if not {'metrics.json','diagnostics.json','trajectory.npz'}<=set(seal['artifacts']): return None
        for name,digest in seal['artifacts'].items():
            path=folder/name
            if not path.resolve().is_relative_to(folder.resolve()) or sha256(path)!=digest: return None
        row=json.loads((folder/'metrics.json').read_text()); check_row(row)
        if row['scene_id']!=sequence or row['frame_ids']!=ids: return None
        return row
    except (OSError,ValueError,KeyError,TypeError): return None


class RssSampler:
    """Per-sequence Linux RSS peak sampled every 100 ms, plus entry/exit."""
    def __init__(self):
        self.peak=0; self.stop=threading.Event()
    def sample(self):
        line=next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:'))
        self.peak=max(self.peak,int(line.split()[1])*1024)
    def poll(self):
        while not self.stop.wait(.1): self.sample()
    def __enter__(self):
        self.sample(); self.thread=threading.Thread(target=self.poll,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.sample(); self.stop.set(); self.thread.join()
