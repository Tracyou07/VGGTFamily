"""Direct fixed 100-frame, two-mode H20 diagnostic launcher."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from .runtime import ROOT, prepare_inputs, preflight, fresh_directory, write_json, source_identity, data_identity
from .windows import make_windows


def validate_config(config):
    expected=(60,30,2,100)
    actual=tuple(config[k] for k in ('window_size','overlap','window_batch_size','diagnostic_frames'))
    if actual!=expected:
        raise ValueError(f'expected fixed 60/30, batch 2, 100 frames; got {actual}')
    if make_windows(100,60,30)!=[(0,60),(30,90),(60,100)]:
        raise ValueError('unexpected fixed window schedule')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['diagnose-direct'])
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/v5_validation.json').read_text())
    validate_config(config)
    if not args.gpu.isdigit():parser.error('--gpu must be physical index')
    identity=source_identity()
    if not identity['commit'] or identity['status']:
        raise RuntimeError('ours_v5 needs a committed, clean working tree')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    lock=open('/tmp/ours_v5_gpu_'+args.gpu+'.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sampler=None;sample_log=None
    try:
        info=preflight(args.gpu,args.output)
        fresh_directory(args.output)
        try:
            write_json(args.output/'preflight.json',info)
            scene=Path(config['scene_root'])
            frame_list=ROOT/'configs/scene0150_00_frames100.json'
            contract=dict(code=identity,data=data_identity(scene,frame_list),checkpoint_sha256=config['checkpoint_sha256'],
                          window_size=60,overlap=30,window_batch_size=2,frames=100,precision='bf16')
            saved=prepare_inputs(scene,frame_list,args.output/'inputs.pt',100)
            write_json(args.output/'config.json',dict(base=config,action=args.action,frame_ids=saved['frame_ids'],
                                                      preprocessing=saved['preprocessing'],contract=contract))
            del saved
            sample_log=(args.output/'vram.csv').open('x')
            sampler=subprocess.Popen(['nvidia-smi','-i',args.gpu,'--query-gpu=timestamp,uuid,memory.used,memory.free,utilization.gpu',
                                      '--format=csv','--loop-ms=500'],stdout=sample_log,stderr=subprocess.STDOUT)
            for mode in ('independent','camera_exchange'):
                command=[sys.executable,'-B','-u','-m','experiments.ours_v5.worker',
                         '--input',str(args.output/'inputs.pt'),'--output',str(args.output/mode),
                         '--gpu',args.gpu,'--frames','100','--mode',mode,'--batch-size','2','--window-size','60','--overlap','30']
                print('START',mode,flush=True)
                with (args.output/(mode+'.log')).open('x') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                if not (args.output/mode/'COMPLETE.json').is_file():
                    raise RuntimeError('worker missing COMPLETE: '+mode)
            summary={}
            for mode in ('independent','camera_exchange'):
                manifest=json.loads((args.output/mode/'run_manifest.json').read_text())
                metric=json.loads((args.output/mode/'trajectory_metrics.json').read_text())
                summary[mode]=dict(ate_rmse_m=metric['ate_rmse_m'],timing=manifest['timing'],
                                   peak_allocated_bytes=manifest['peak_allocated_bytes'],
                                   peak_reserved_bytes=manifest['peak_reserved_bytes'],
                                   cpu_peak_rss_bytes=manifest['cpu_peak_rss_bytes'],gpu_uuid=manifest['gpu_uuid'])
            write_json(args.output/'summary.json',summary)
            write_json(args.output/'COMPLETE.json',dict(status='complete',action=args.action,modes=list(summary)))
        except Exception as error:
            write_json(args.output/'FAILED.json',dict(reason=str(error),traceback=traceback.format_exc()))
            raise
    finally:
        if sampler is not None:sampler.terminate();sampler.wait(timeout=10)
        if sample_log is not None:sample_log.close()
        lock.close()


if __name__=='__main__':main()
