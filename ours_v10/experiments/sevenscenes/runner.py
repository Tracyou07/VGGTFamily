"""7Scenes adapter: prediction-only v4 reconstruction, then the shared protocol."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from .common import frames_for_sequence,point_maps,sha256,summarize,write_json,seal_attempt,valid_attempt,RssSampler
from .engine import infer_windows,stitch_predictions,compare_predictions
from experiments.ours_v3.geometry import AlignmentConfig

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT_SHA='f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
TOLERANCE=dict(atol=.02,rtol=.02,center_m=.01,rotation_deg=.5)


def gpu_preflight(gpu,output_parent,min_free_gib=10):
    if not str(gpu).isdigit(): raise ValueError('GPU_ID must be a physical integer index')
    if socket.gethostname()!='VM-0-11-ubuntu' or os.environ.get('USER')!='ubuntu':
        raise ValueError('GPU work is authorized only on the configured H20 / ubuntu')
    line=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    index,uuid,memory,util=[s.strip() for s in line.split(',')]
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
    active=[line for line in jobs.splitlines() if line.split(',')[0].strip()==uuid]
    if float(memory)>256 or float(util)>5 or active: raise RuntimeError(f'GPU {gpu} busy: {line}; jobs={active}')
    path=Path(output_parent)
    while not path.exists(): path=path.parent
    free=shutil.disk_usage(path).free
    if free<min_free_gib*2**30: raise RuntimeError('insufficient output disk space (minimum 10 GiB)')
    return dict(host=socket.gethostname(),user=os.environ.get('USER'),gpu_id=index,gpu_uuid=uuid,
                memory_used_mib=float(memory),utilization=float(util),active_jobs=active,output_disk_free_bytes=free)


def source_identity(eval_root):
    paths=[]
    for sub in ('vggt','experiments/ours_v3','experiments/ours_v4','experiments/sevenscenes'):
        paths.extend((ROOT/sub).rglob('*.py'))
    paths.extend((eval_root/'reference/FastVGGT-main/eval').rglob('*.py'))
    paths += [eval_root/'adapters/registered_data.py',eval_root/'reference/FastVGGT-main/vggt/utils/eval_utils.py']
    return {str(p):sha256(p) for p in sorted(set(paths))}


def load_model(checkpoint):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    import vggt.models.vggt as implementation
    if not Path(implementation.__file__).resolve().is_relative_to(ROOT): raise RuntimeError('wrong VGGT implementation')
    model=VGGT().eval().requires_grad_(False)
    state=load_file(str(checkpoint)); model.load_state_dict(state,strict=True); del state
    return model.cuda()


def prepare_images(paths):
    from vggt.utils.load_fn import load_and_preprocess_images
    torch.cuda.synchronize(); start=time.perf_counter()
    images=load_and_preprocess_images([str(p) for p in paths])
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    if tuple(images.shape[1:])!=(3,392,518): raise ValueError('protocol requires 518 x 392 images')
    return images,elapsed


def run_gate(model,args,dataset,manifest):
    sequence=args.sequence or dataset.scene_list[0]
    ids,paths=frames_for_sequence(args.data_root,sequence,args.kf)
    if len(ids)<55: raise ValueError('gate needs at least 55 sampled frames')
    ids=ids[:55]; paths=paths[:55]; images,prep=prepare_images(paths)
    records={}; predictions={}; global_results={}
    # Budgets are in the manifest before execution, unchanged from v4's BF16 gate.
    for batch in (1,2):
        folder=args.output_dir/f'batch{batch}'; folder.mkdir()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        local,windows,groups,timing=infer_windows(model,images,ids,batch)
        global_result,stitch_seconds,export_seconds=stitch_predictions(local,ids,folder)
        for index,pred in enumerate(local): np.savez_compressed(folder/f'window_{index:04}.npz',**pred)
        np.savez_compressed(folder/'global_trajectory.npz',frame_ids=global_result['frame_ids'],c2w=global_result['c2w'],intrinsics=global_result['intrinsics'])
        records[str(batch)]=dict(windows=windows,packed_groups=groups,timing=timing,preprocessing_seconds=prep,
            stitching_seconds=stitch_seconds,diagnostic_export_seconds=export_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        predictions[batch]=local; global_results[batch]=global_result
    local_report=compare_predictions(predictions[1],predictions[2],TOLERANCE)
    stitched_report=compare_predictions([global_results[1]],[global_results[2]],TOLERANCE)
    report=dict(local=local_report,stitched=stitched_report,resources=records,scene_id=sequence,frame_ids=ids,
                gpu_uuid=manifest['preflight']['gpu_uuid'],passed=local_report['passed'] and stitched_report['passed'])
    write_json(args.output_dir/'gate_report.json',report)
    if not report['passed']: raise RuntimeError('7Scenes BF16 equivalence gate failed; stop before smoke')
    write_json(args.output_dir/'GATE_COMPLETE.json',dict(passed=True,fingerprint=manifest['fingerprint']))


def run_sequence(model,args,dataset,index,ids,paths,attempt):
    torch.cuda.reset_peak_memory_stats()
    images,prep=prepare_images(paths)
    input_hash=hashlib.sha256(images.numpy().tobytes()).hexdigest()
    local,windows,groups,timing=infer_windows(model,images,ids,2)
    torch.cuda.synchronize()
    result,stitch_seconds,export_seconds=stitch_predictions(local,ids,attempt)
    torch.cuda.synchronize(); start=time.perf_counter()
    points=point_maps(result['depth'],result['intrinsics'],result['c2w'])
    torch.cuda.synchronize(); map_seconds=time.perf_counter()-start
    reconstruction_total=prep+sum(timing.values())+stitch_seconds+map_seconds
    reconstruction_peak=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    # Export only compact local cameras. Dense gate predictions are kept by run_gate.
    for window,(lo,hi) in enumerate(windows):
        folder=attempt/'windows'/f'{window:04}'; folder.mkdir(parents=True)
        pred=local[window]
        np.savez_compressed(folder/'cameras.npz',frame_ids=pred['frame_ids'],c2w=pred['c2w'],intrinsics=pred['intrinsics'])
    np.savez_compressed(attempt/'trajectory.npz',frame_ids=result['frame_ids'],c2w=result['c2w'],
                        intrinsics=result['intrinsics'],source_window=result['source_window'])
    del local,images
    # GT is first loaded here, after inference, Sim3 and ownership have finished.
    from .protocol import evaluate_scene,bootstrap
    from torch.utils.data._utils.collate import default_collate
    _,criterion=bootstrap(args.eval_root)
    torch.cuda.synchronize(); start=time.perf_counter()
    gt=default_collate([dataset[index]])
    actual=[Path(v['instance'][0]) for v in gt]
    if actual!=paths: raise ValueError('GT evaluator frame list differs from inference frame list')
    row=evaluate_scene(gt,points,result['confidence'],criterion)
    torch.cuda.synchronize(); evaluation_seconds=time.perf_counter()-start
    row.update(mean_nc=(row['nc1']+row['nc2'])/2,scene_id=dataset.scene_list[index],frame_ids=ids,
        frames=len(ids),kf=args.kf,point_map='depth_unprojection',gpu_uuid=args.gpu_uuid,
        timings=dict(image_preprocessing=prep,**timing,overlap_stitching=stitch_seconds,
            point_map_unprojection=map_seconds,reconstruction_total=reconstruction_total,
            protocol_evaluation=evaluation_seconds,stitch_diagnostic_export=export_seconds),
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**reconstruction_peak)
    diagnostics=dict(frame_ids=ids,relative_rgb_paths=[str(p.relative_to(args.data_root)) for p in paths],
        windows=windows,packed_groups=groups,window_frame_ids=[ids[lo:hi] for lo,hi in windows],
        source_window=result['source_window'].tolist(),image_tensor_sha256=input_hash,
        preprocessing=dict(loader='original VGGT* default crop',shape=[len(ids),3,392,518],dtype='float32',range=[0,1]),
        timings=row['timings'],alignment=asdict(AlignmentConfig()),coordinate_convention='depth=camera z; c2w rotation proper; depth and center scaled, intrinsics unchanged',
        gpu_uuid=args.gpu_uuid,window_batch_size=2)
    write_json(attempt/'diagnostics.json',diagnostics); write_json(attempt/'metrics.json',row)
    return row


def run(args):
    from .protocol import bootstrap
    # Import order binds the frozen model before the external evaluator dependencies.
    import vggt.models.vggt
    SevenScenes,_=bootstrap(args.eval_root)
    from registered_data import validate_registered_root
    preflight=gpu_preflight(args.gpu_id,args.output_dir.parent)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu_id): raise ValueError('CUDA_VISIBLE_DEVICES must match --gpu-id')
    registration=validate_registered_root(args.data_root)
    dataset=SevenScenes(split='test',ROOT=str(args.data_root),resolution=(518,392),num_seq=1,full_video=True,kf_every=args.kf)
    expected=list(dataset.scene_list)
    if len(expected)!=18 or len(set(expected))!=18: raise ValueError('expected exactly 18 test sequences')
    if args.sequence and args.sequence not in expected: raise ValueError('sequence is not in test split')
    selected=[args.sequence] if args.sequence else expected
    frame_lists={s:frames_for_sequence(args.data_root,s,args.kf)[0] for s in expected}
    metadata=[]
    for sequence in selected:
        for frame in frame_lists[sequence]:
            for suffix in ('color.png','pose.txt','depth.proj.png'):
                p=args.data_root/sequence/f'frame-{frame}.{suffix}'; stat=p.stat()
                metadata.append((str(p),stat.st_size,stat.st_mtime_ns))
    checkpoint_hash=sha256(args.checkpoint)
    if checkpoint_hash!=CHECKPOINT_SHA: raise ValueError('checkpoint SHA256 mismatch')
    contract=dict(kf=args.kf,selected_sequences=selected,expected_sequences=expected,frame_lists=frame_lists,
        data_root=str(args.data_root),data_stat_sha256=hashlib.sha256(json.dumps(metadata).encode()).hexdigest(),
        registration=registration,checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_hash,
        sources=source_identity(args.eval_root),window_size=30,overlap=10,window_batch_size=2,
        alignment=asdict(AlignmentConfig()),precision='BF16 autocast; original heads',gate_only=args.gate_only,
        tolerance=TOLERANCE,seed=2026,point_map='depth_unprojection',comparison_note='VGGT-Long uses point-head world_points; not a batching-only ablation',
        protocol='existing FastVGGT 7Scenes; Regr3D_t_ScaleShiftInv(L21,norm_mode=False,gt_scale=True); center224; p2p ICP 0.1m',
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,attention_backend='original SDPA dispatcher',omp_threads=os.environ.get('OMP_NUM_THREADS'))
    fingerprint=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    manifest=dict(contract=contract,fingerprint=fingerprint,preflight=preflight,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        start_commit='a3b13e561de97b0699e42f1cd29fe0918b5483ce',working_tree=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
    manifest_path=args.output_dir/'run_manifest.json'
    if args.resume:
        old=json.loads(manifest_path.read_text())
        if old['fingerprint']!=fingerprint: raise ValueError('resume provenance mismatch; use a new run directory')
    else:
        if manifest_path.exists() or (args.output_dir.exists() and any(p.name not in ('raw.log','vram.csv','preflight.json') for p in args.output_dir.iterdir())):
            raise ValueError('output already contains a run; never overwrite')
        args.output_dir.mkdir(parents=True,exist_ok=True); write_json(manifest_path,manifest)
    args.gpu_uuid=preflight['gpu_uuid']
    rows=[]; pending=[]
    for sequence in selected:
        parent=args.output_dir/'sequences'/sequence.replace('/','__'); row=None
        if args.resume:
            for attempt in sorted(parent.glob('attempt_*'),reverse=True):
                row=valid_attempt(attempt,fingerprint,sequence,frame_lists[sequence])
                if row is not None: print('SKIP verified',sequence,flush=True); break
        if row is None: pending.append(sequence)
        else: rows.append(row)
    def update_summary():
        rows.sort(key=lambda r:expected.index(r['scene_id']))
        temp=args.output_dir/'sequences.jsonl.tmp'
        temp.write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows)); temp.replace(args.output_dir/'sequences.jsonl')
        summary=summarize(rows,expected); summary.update(kf=args.kf,selected_sequences=selected,fingerprint=fingerprint)
        write_json(args.output_dir/'summary.json',summary)
        return summary
    update_summary()
    if args.resume and pending:
        for name in ('COMPLETE.json','SMOKE_COMPLETE.json'):
            marker=args.output_dir/name
            if marker.exists(): marker.rename(marker.with_name(marker.stem+f'.previous_{time.time_ns()}.json'))
    try:
        torch.manual_seed(2026); np.random.seed(2026)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        model=None
        if pending or args.gate_only:
            gpu_preflight(args.gpu_id,args.output_dir.parent)
            model=load_model(args.checkpoint)
        if args.gate_only:
            if args.resume: raise ValueError('gate resume is not supported; use new output')
            run_gate(model,args,dataset,manifest); return
        for sequence in pending:
            parent=args.output_dir/'sequences'/sequence.replace('/','__'); parent.mkdir(parents=True,exist_ok=True)
            numbers=[int(p.name.split('_')[-1]) for p in parent.glob('attempt_*')]
            attempt=parent/f'attempt_{max(numbers,default=0)+1:04}'; attempt.mkdir()
            print('START',sequence,str(attempt),flush=True)
            try:
                ids,paths=frames_for_sequence(args.data_root,sequence,args.kf)
                with RssSampler() as rss:
                    row=run_sequence(model,args,dataset,dataset.scene_list.index(sequence),ids,paths,attempt)
                row['cpu_process_high_water_rss_bytes']=row['cpu_peak_rss_bytes']
                row['cpu_peak_rss_bytes']=rss.peak; row['cpu_rss_sampling_interval_seconds']=.1
                write_json(attempt/'metrics.json',row)
                seal_attempt(attempt,fingerprint,sequence,ids)
                if valid_attempt(attempt,fingerprint,sequence,ids) is None: raise ValueError('post-write integrity failed')
                rows.append(row); update_summary(); print(json.dumps(row),flush=True)
            except Exception as error:
                write_json(attempt/'FAILED.json',dict(reason=str(error),traceback=traceback.format_exc())); raise
        summary=update_summary()
        marker='COMPLETE.json' if summary['complete'] else 'SMOKE_COMPLETE.json'
        write_json(args.output_dir/marker,dict(valid_sequences=len(rows),expected_sequences=18,formal_complete=summary['complete'],fingerprint=fingerprint))
        if (args.output_dir/'FAILED.json').exists():
            # Keep history; successful recovery never erases the original traceback.
            write_json(args.output_dir/'RECOVERED.json',dict(fingerprint=fingerprint,valid_sequences=len(rows)))
    except Exception as error:
        update_summary()
        write_json(args.output_dir/'FAILED.json',dict(reason=str(error),traceback=traceback.format_exc()))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kf',type=int,choices=(3,10),required=True); p.add_argument('--gpu-id',required=True)
    p.add_argument('--data-root',type=Path,default='/data/yjh/share/datasets/7scenes_registered_simplerecon_v1')
    p.add_argument('--checkpoint',type=Path,default='/data/yjh/share/pretrained/VGGT-1B/model.safetensors')
    p.add_argument('--eval-root',type=Path,default=ROOT.parent/'eval/7scenes')
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--sequence')
    p.add_argument('--resume',action='store_true'); p.add_argument('--gate-only',action='store_true')
    args=p.parse_args()
    try: run(args)
    except Exception as error:
        # Preflight/provenance rejections must not alter an existing run.
        if args.output_dir.exists() and not (args.output_dir/'run_manifest.json').exists():
            write_json(args.output_dir/'FAILED.json',dict(reason=str(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__': main()
