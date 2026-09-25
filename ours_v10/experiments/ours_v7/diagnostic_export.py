"""Opt-in lossless NPZ compression level 1; default inference export unchanged."""
import argparse
import io
from pathlib import Path
import time
import zipfile
import numpy as np
from experiments.ours_v7.diagnostics import OLD,csv_rows,write_json,tensor_hash

def save_npz_fast(file, **arrays):
    with zipfile.ZipFile(file,mode='w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,value in arrays.items():
            with z.open(name+'.npy','w',force_zip64=True) as entry:
                np.lib.format.write_array(entry,np.asanyarray(value),allow_pickle=False)

def benchmark(out,repeats=2):
    rows=[];checks=[]
    sources=[OLD/'independent/windows'/f'{w:04d}'/'local.npz' for w in range(3)]
    sources.append(OLD/'independent/global_predictions.npz')
    for p in sources:
        with np.load(p,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        original={k:tensor_hash(v) for k,v in values.items()}
        for iteration in range(repeats):
            # Reverse order on second pass to reduce fixed order bias.
            levels=[6,1] if iteration%2==0 else [1,6]
            for level in levels:
                stream=io.BytesIO();t=time.perf_counter()
                if level==6:np.savez_compressed(stream,**values)
                else:save_npz_fast(stream,**values)
                elapsed=time.perf_counter()-t
                n=stream.tell();stream.seek(0)
                with np.load(stream,allow_pickle=False) as restored:
                    exact=all(tensor_hash(restored[k])==original[k] for k in values)
                checks.append(dict(source=str(p),iteration=iteration,level=level,all_arrays_exact=exact))
                assert exact
                rows.append(dict(stage='npz_serialization',source=p.name if p.name!='local.npz' else p.parent.name,
                    iteration=iteration,level=level,seconds=elapsed,bytes=n,cold=iteration==0,
                    scope='complete original arrays serialized to RAM; filesystem write measured separately',exact=exact))
                if iteration==0:
                    # Real fresh-file write+fsync, identical destination device, then remove only this owned temporary file.
                    dest=out/f'export_probe_{p.parent.name}_{p.stem}_level{level}.npz'
                    assert not dest.exists()
                    import os
                    t=time.perf_counter()
                    with open(dest,'xb') as f:
                        f.write(stream.getbuffer());f.flush();os.fsync(f.fileno())
                    disk=time.perf_counter()-t
                    rows.append(dict(stage='write_and_fsync',source=p.name if p.name!='local.npz' else p.parent.name,
                        iteration=iteration,level=level,seconds=disk,bytes=n,cold=True,
                        scope='fresh file on output filesystem; removed after timing',exact=True))
                    dest.unlink()
                stream.close()
                csv_rows(out/'export_optimization.csv',rows);write_json(out/'export_regression.json',checks)
                print(p.parent.name,p.name,iteration,level,elapsed,n,exact,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2);a=p.parse_args();benchmark(a.output,a.repeats)
