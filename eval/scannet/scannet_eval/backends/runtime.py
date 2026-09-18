"""Upstream orchestration. Native source trees are imported, never edited."""
from contextlib import contextmanager
from pathlib import Path
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
from .common import (load_state, checked_load, validate_scene, w2c_to_c2w, finish_prediction, stage_images, extract_long_chunks, extract_slam_submaps)


def install_source(root, namespace, package_root=None):
    root=Path(root).resolve(); expected=Path(package_root or root/namespace).resolve()
    for name,module in list(sys.modules.items()):
        if name==namespace or name.startswith(namespace+'.'):
            filename=getattr(module,'__file__',None)
            if filename and not Path(filename).resolve().is_relative_to(expected):
                raise RuntimeError(f'namespace collision: {name} from {filename}; run each backend in a fresh process')
    sys.path.insert(0,str(root)); importlib.invalidate_caches()


def as_numpy(value):
    return value.detach().float().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def depth_points(points, depth, confidence, threshold):
    points=np.asarray(points); depth=np.asarray(depth); confidence=np.asarray(confidence)
    if depth.shape==points.shape[:-1]+(1,): depth=depth[...,0]
    if points.shape[:-1]!=depth.shape or confidence.shape!=depth.shape: raise ValueError('depth/points/confidence shapes disagree')
    mask=np.isfinite(depth)&(depth>0)&np.isfinite(confidence)&(confidence>=threshold)
    return points[mask], {'confidence_rule':'depth_conf >= threshold, finite confidence and positive finite depth','depth_conf_thresh':float(threshold),'depth_or_confidence_removed':int((~mask).sum())}


@contextmanager
def offline_hub(source, mode, torch_home=None):
    """Redirect only the exact native DINO request; disallow accidental downloads."""
    import torch
    old=torch.hub.load; old_home=os.environ.get('TORCH_HOME'); old_safe=os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
    if torch_home: os.environ['TORCH_HOME']=str(torch_home)
    def load(repo_or_dir, model, *args, **kwargs):
        if mode=='long' and repo_or_dir=='./LoopModels/dinov2':
            repo_or_dir=str(Path(source)/'LoopModels/dinov2'); kwargs['source']='local'; kwargs['pretrained']=False
        elif mode=='slam' and repo_or_dir=='facebookresearch/dinov2':
            repo_or_dir=str(Path(torch_home)/'hub/facebookresearch_dinov2_main'); kwargs['source']='local'; kwargs['pretrained']=False
        elif kwargs.get('source')!='local':
            raise RuntimeError(f'unexpected online torch.hub request: {repo_or_dir}')
        return old(repo_or_dir,model,*args,**kwargs)
    torch.hub.load=load
    try: yield
    finally:
        torch.hub.load=old
        if old_safe is None: os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD',None)
        else: os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']=old_safe
        if old_home is None: os.environ.pop('TORCH_HOME',None)
        else: os.environ['TORCH_HOME']=old_home


class NativeBackend:
    def __init__(self,name,config,device): self.name=name; self.config=config; self.device=str(device)

    def predict(self,scene,work_dir):
        ids,paths=validate_scene(scene)
        work_dir=Path(work_dir)
        if not work_dir.is_absolute(): raise ValueError('work_dir must be absolute')
        work_dir.mkdir(parents=True,exist_ok=True)
        from . import doctor_backend
        diagnosis=doctor_backend(self.name,self.config)
        if not diagnosis['ready']: raise RuntimeError('; '.join(diagnosis['errors']))
        import torch
        if not self.device.startswith('cuda'): raise ValueError('native inference requires CUDA; CPU doctor/tests do not allocate models')
        cuda_device=torch.device(self.device)
        cuda_index=torch.cuda.current_device() if cuda_device.index is None else cuda_device.index
        torch.cuda.set_device(cuda_index); self.device=f'cuda:{cuda_index}'; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        # Native VGGT implementations assume the current CUDA device.
        os.environ['PYTHONDONTWRITEBYTECODE']='1'; sys.dont_write_bytecode=True
        methods={'vggt_original':self._vggt,'vggt_star':self._vggt,'fastvggt':self._vggt,'streamvggt':self._stream,'omega':self._omega,'long':self._long,'slam':self._slam}
        points,poses,seconds,meta=methods[self.name](ids,paths,work_dir)
        torch.cuda.synchronize()
        allocated=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
        root=Path(self.config['project_root'])
        revision=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],text=True,capture_output=True)
        meta.update(model=self.name,source_root=str(root),source_revision=revision.stdout.strip() if revision.returncode==0 else None,checkpoint=self.config['checkpoint'],config=self.config,python=sys.executable,peak_memory_scope='model construction through native inference and output extraction',parameter_dtype='bfloat16' if self.name=='fastvggt' else 'float32',autocast_dtype='bfloat16')
        return finish_prediction(points,poses,ids,seconds,allocated,reserved,meta,int(self.config['max_points']))

    def _vggt(self,ids,paths,work_dir):
        import torch
        root=self.config['project_root']; install_source(root,'vggt')
        from vggt.models.vggt import VGGT
        fast=self.name=='fastvggt'
        kwargs=dict(merging=int(self.config['merging']),merge_ratio=float(self.config['merge_ratio'])) if fast else {}
        model=VGGT(**kwargs)
        weights=checked_load(model,load_state(self.config['checkpoint']),('point_head.','track_head.') if fast else ())
        model=model.eval().to(self.device)
        if fast:
            from PIL import Image
            from scannet_eval.vendor.fastvggt_eval_utils import get_vgg_input_imgs,infer_vggt_and_reconstruct
            images=[np.asarray(Image.open(path).convert('RGB')) for path in paths]
            images,pw,ph=get_vgg_input_imgs(images)
            model.update_patch_dimensions(pw,ph)
            model=model.to(torch.bfloat16)
            with torch.inference_mode():
                extrinsic,_,clouds,_,_,milliseconds=infer_vggt_and_reconstruct(model,images,torch.bfloat16,float(self.config['depth_conf_thresh']),image_paths=paths)
            return np.concatenate(clouds),w2c_to_c2w(extrinsic),milliseconds/1000,dict(weights=weights,camera_decode_dtype='float32',confidence_rule='vendored FastVGGT depth_conf >= threshold; finite points',depth_conf_thresh=float(self.config['depth_conf_thresh']),timing_scope='vendored synchronized model forward, including input transfer')
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        images=load_and_preprocess_images(paths).to(self.device)
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): predictions=model(images)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        extrinsic,intrinsic=pose_encoding_to_extri_intri(predictions['pose_enc'],images.shape[-2:])
        ext=as_numpy(extrinsic)[0]; intr=as_numpy(intrinsic)[0]; depth=as_numpy(predictions['depth'])[0]; conf=as_numpy(predictions['depth_conf'])[0]
        points=unproject_depth_map_to_point_map(depth,ext,intr)
        points,meta=depth_points(points,depth,conf,float(self.config['depth_conf_thresh']))
        meta.update(weights=weights,timing_scope='synchronized native model forward')
        return points,w2c_to_c2w(ext),seconds,meta

    def _stream(self,ids,paths,work_dir):
        import torch
        install_source(self.config['project_root'],'streamvggt')
        from streamvggt.models.streamvggt import StreamVGGT
        from streamvggt.utils.load_fn import load_and_preprocess_images
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
        from streamvggt.utils.geometry import unproject_depth_map_to_point_map
        model=StreamVGGT(); weights=checked_load(model,load_state(self.config['checkpoint'])); model=model.eval().to(self.device)
        images=load_and_preprocess_images(paths).to(self.device)
        frames=[{'img':image.unsqueeze(0)} for image in images]
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): result=model.inference(frames)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        if len(result.ress)!=len(ids): raise ValueError('Stream native inference omitted frames')
        poses=torch.stack([r['camera_pose'] for r in result.ress],dim=1)
        depth=as_numpy(torch.stack([r['depth'] for r in result.ress],dim=1))[0]
        conf=as_numpy(torch.stack([r['depth_conf'] for r in result.ress],dim=1))[0]
        extrinsic,intrinsic=pose_encoding_to_extri_intri(poses,images.shape[-2:]); ext=as_numpy(extrinsic)[0]; intr=as_numpy(intrinsic)[0]
        points,meta=depth_points(unproject_depth_map_to_point_map(depth,ext,intr),depth,conf,float(self.config['depth_conf_thresh']))
        meta.update(weights=weights,native_api='StreamVGGT.inference sequential KV cache',timing_scope='synchronized native sequential inference')
        return points,w2c_to_c2w(ext),seconds,meta

    def _omega(self,ids,paths,work_dir):
        import torch
        root=Path(self.config['project_root']); install_source(root,'vggt_omega')
        spec=importlib.util.spec_from_file_location('_scannet_native_omega_run',root/'run.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        # Local facade keeps external load_model intact while selecting a safe
        # deserializer. It does not change torch.load globally.
        class SafeTorch:
            def __getattr__(self,key): return getattr(torch,key)
            def load(self,path,*args,**kwargs):
                if Path(path).resolve()!=Path(self_checkpoint).resolve(): raise ValueError('unexpected Omega checkpoint request')
                return load_state(path)
        self_checkpoint=self.config['checkpoint']; module.torch=SafeTorch()
        model=module.load_model(self_checkpoint,self.device)
        torch.cuda.synchronize(); start=time.perf_counter()
        predictions=module.run_inference(model,paths,int(self.config['image_size']),self.config['image_mode'],int(self.config['patch_size']),self.device)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
        points,meta=depth_points(predictions['world_points_from_depth'],predictions['depth'],predictions['depth_conf'],float(self.config['depth_conf_thresh']))
        meta.update(native_api='external run.py load_model / run_inference',weights={'strict':True,'safe_deserializer':True},timing_scope='native run_inference including preprocessing, camera decode and backprojection')
        return points,w2c_to_c2w(predictions['extrinsic']),seconds,meta

    def _long(self,ids,paths,work_dir):
        import torch
        import yaml
        root=Path(self.config['project_root']); install_source(root,'base_models'); install_source(root/'base_models','vggt')
        sys.path.insert(0,self.config['dependency_path'])
        import vggt_long
        from base_models.base_model import VGGTAdapter
        from base_models.vggt.models.vggt import VGGT
        config=yaml.safe_load((root/'configs/base_config.yaml').read_text())
        config['Weights'].update(model='VGGT',VGGT=self.config['checkpoint'],SALAD=self.config['salad_checkpoint'],DNIO=self.config['dino_checkpoint'])
        config['Model'].update(chunk_size=int(self.config['chunk_size']),overlap=int(self.config['overlap']),loop_chunk_size=int(self.config['loop_chunk_size']),loop_enable=True,useDBoW=False,using_sim3=True,reference_frame_mid=False,calib=False,delete_temp_files=False)
        config['Model']['IRLS']['tol']='1e-9'
        config['Model']['Pointcloud_Save'].update(use_conf_filter=True,conf_threshold_coef=float(self.config['conf_threshold_coef']))
        config['Loop']['SIM3_Optimizer']['lang_version']='python'; config['Loop']['SALAD']['batch_size']=int(self.config['salad_batch_size'])
        staged,_=stage_images(ids,paths,work_dir); output=work_dir/'native_long'; output.mkdir(exist_ok=False)
        weights={}
        class SafeAdapter(VGGTAdapter):
            def load(adapter):
                adapter.model=VGGT(); weights.update(checked_load(adapter.model,load_state(adapter.config['Weights']['VGGT']))); adapter.model=adapter.model.eval().to(adapter.device)
        with offline_hub(root,'long'):
            native=vggt_long.VGGT_Long(str(staged),str(output),config)
            native.model=SafeAdapter(config,device=self.device)
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.inference_mode(): native.run()
            torch.cuda.synchronize(); seconds=time.perf_counter()-start
        # These dictionaries were just emitted into a new run-only directory by
        # the native pipeline. No user-supplied pickled arrays are accepted.
        chunks=[np.load(Path(native.result_unaligned_dir)/f'chunk_{k}.npy',allow_pickle=True).item() for k in range(len(native.chunk_indices))]
        points,poses,meta=extract_long_chunks(chunks,native.chunk_indices,native.sim3_list,len(ids),float(self.config['conf_threshold_coef']))
        exported=np.loadtxt(output/'camera_poses.txt').reshape(-1,4,4)
        if not np.allclose(poses,exported,atol=1e-4): raise ValueError('native Long exported cameras disagree with chunk ownership')
        meta.update(weights=weights,resolved_native_config=config,native_api='VGGT_Long.run',loop_pairs=len(native.loop_list),loop_constraints=len(native.loop_sim3_list),timing_scope='native run including retrieval, model load, chunk inference, alignment, loop optimizer and exports',point_source='unaligned native dictionaries plus final cumulative Sim3; native PLY duplication avoided')
        (work_dir/'native_long_config.json').write_text(json.dumps(config,indent=2))
        # Temporary arrays are needed during native Long processing, but final
        # points and poses are already materialized and checked at this point.
        temporary_dirs = [
            Path(native.result_unaligned_dir),
            Path(native.result_aligned_dir),
            Path(native.result_loop_dir),
        ]
        cleanup_errors = []
        for temporary_dir in temporary_dirs:
            try:
                if temporary_dir.exists():
                    shutil.rmtree(temporary_dir)
            except OSError as error:
                cleanup_errors.append(f'{temporary_dir}: {error}')
        meta['temporary_artifacts_removed'] = not cleanup_errors
        if cleanup_errors:
            meta['temporary_artifact_cleanup_errors'] = cleanup_errors
        return points,poses,seconds,meta

    def _slam(self,ids,paths,work_dir):
        import torch
        root=Path(self.config['project_root']); install_source(root,'vggt_slam'); install_source(root/'third_party/salad','salad'); install_source(root/'third_party/vggt','vggt')
        import vggt_slam.solver as solver_module
        import vggt.utils.geometry as geometry
        from vggt.models.vggt import VGGT
        expected=root/'third_party/vggt'
        if not Path(geometry.__file__).resolve().is_relative_to(expected.resolve()): raise RuntimeError('SLAM requires native camera-local geometry')
        class NullViewer:
            def __init__(self,*args,**kwargs): pass
        staged,staged_paths=stage_images(ids,paths,work_dir)
        path_ids=dict(zip(staged_paths,ids)); model=VGGT(); weights=checked_load(model,load_state(self.config['checkpoint'])); model=model.eval().to(self.device)
        class InferenceModel:
            calls=0
            def __call__(wrapper,*args,**kwargs):
                wrapper.calls+=1
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): return model(*args,**kwargs)
        wrapped=InferenceModel(); old_viewer=solver_module.Viewer; solver_module.Viewer=NullViewer
        try:
            with offline_hub(root,'slam',self.config['torch_home']):
                solver=solver_module.Solver(init_conf_threshold=float(self.config['conf_percentile']),lc_thres=float(self.config['lc_thres']))
                torch.cuda.synchronize(); start=time.perf_counter(); optimize_calls=0
                for offset in range(0,len(staged_paths),int(self.config['submap_size'])):
                    window=staged_paths[offset:offset+int(self.config['submap_size'])+1]
                    if len(window)==1 and offset>0: continue
                    predictions=solver.run_predictions(window,wrapped,max_loops=int(self.config['max_loops']),clip_model=None,clip_preprocess=None)
                    solver.add_points(predictions); solver.graph.optimize(); optimize_calls+=1
                torch.cuda.synchronize(); seconds=time.perf_counter()-start
        finally: solver_module.Viewer=old_viewer
        points,poses,meta=extract_slam_submaps(solver.map.ordered_submaps_by_key(),solver.graph,path_ids,ids)
        meta.update(weights=weights,native_api='Solver.run_predictions / add_points / graph.optimize',forward_calls=wrapped.calls,graph_optimize_calls=optimize_calls,accepted_loop_closures=solver.graph.get_num_loops(),conf_percentile=float(self.config['conf_percentile']),depth_geometry='native camera-local maps transformed by optimized per-frame SL4; poses from native RQ decomposition',timing_scope='native submap preprocessing, retrieval, forwards, graph additions and optimization; excludes model construction',dino_policy='offline local torch.hub with pretrained=False; complete native SALAD checkpoint then loads strictly')
        return points,poses,seconds,meta
