"""Seven isolated native inference backends for the FastVGGT ScanNet protocol.

Run each backend in a fresh process: several upstreams own the same ``vggt``
namespace. Construction and doctor are allocation-free; predict loads models.
"""
from pathlib import Path
import importlib.util
import sys
from .common import Prediction

BASE = Path('/home/ubuntu/yjh/feedforwardreconstruct')
SHARED = Path('/data/yjh/share/pretrained')
ROOTS = {'vggt_original': BASE/'vggtlong/base_models', 'vggt_star': BASE/'vggt', 'fastvggt': BASE/'eval/7scenes/reference/FastVGGT-main', 'streamvggt': BASE/'vggtstream/src', 'long': BASE/'vggtlong', 'slam': BASE/'vggtslam', 'omega': BASE/'vggtomega'}
CHECKPOINTS = {'streamvggt': SHARED/'StreamVGGT/checkpoints.pth', 'omega': SHARED/'VGGT-Omega/vggt_omega_1b_512.pt'}


def resolve_config(name, config):
    if name not in ROOTS: raise ValueError(f'unknown model: {name}')
    c = dict(config)
    defaults = dict(project_root=str(ROOTS[name]), checkpoint=str(CHECKPOINTS.get(name,SHARED/'VGGT-1B/model.safetensors')), image_size=512 if name=='omega' else 518, max_points=0, depth_conf_thresh=1.0, merging=0, merge_ratio=.9, chunk_size=60, overlap=30, loop_chunk_size=20, dependency_path=str(Path(__file__).resolve().parents[2]/'.runtime/long_deps'), salad_checkpoint=str(SHARED/'torch/hub/checkpoints/dino_salad.ckpt'), dino_checkpoint=str(SHARED/'torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'), salad_batch_size=2, conf_threshold_coef=.75, submap_size=16, max_loops=1, conf_percentile=25., lc_thres=.95, torch_home=str(SHARED/'torch'), image_mode='balanced', patch_size=16 if name=='omega' else 14)
    for key,value in defaults.items():
        if c.get(key) is None: c[key]=value
    for key in ('project_root','checkpoint','dependency_path','salad_checkpoint','dino_checkpoint','torch_home'):
        c[key]=str(Path(c[key]).expanduser().resolve())
    if int(c['image_size']) != (512 if name=='omega' else 518): raise ValueError(f'{name} supports the verified native image_size={512 if name=="omega" else 518} profile only')
    if int(c['max_points'])<0: raise ValueError('max_points must be nonnegative')
    if name=='long' and not 0<int(c['overlap'])<int(c['chunk_size']): raise ValueError('Long requires 0 < overlap < chunk_size')
    if name=='slam' and (int(c['max_loops']) not in (0,1) or int(c['submap_size'])<1): raise ValueError('SLAM requires max_loops<=1 and submap_size>=1')
    if name=='slam' and not 0<=float(c['conf_percentile'])<100: raise ValueError('SLAM percentile must be in [0,100)')
    if name=='omega' and (int(c['patch_size'])!=16 or c['image_mode'] not in ('balanced','max_size')): raise ValueError('Omega requires patch_size16 and balanced/max_size mode')
    return c


def doctor_backend(model_name, config):
    errors=[]
    try: c=resolve_config(model_name,config)
    except (ValueError,TypeError) as error: return {'ready':False,'errors':[str(error)],'diagnostics':{}}
    root=Path(c['project_root'])
    required=[Path(c['checkpoint'])]
    modules=['torch','numpy','PIL','safetensors']
    if model_name in ('vggt_original','vggt_star','fastvggt'): required += [root/'vggt/models/vggt.py']
    elif model_name=='streamvggt': required += [root/'streamvggt/models/streamvggt.py']
    elif model_name=='omega': required += [root/'run.py',root/'vggt_omega/models/__init__.py']
    elif model_name=='long':
        required += [root/'vggt_long.py', root/'configs/base_config.yaml',Path(c['salad_checkpoint']),Path(c['dino_checkpoint']),Path(c['dependency_path'])]
        modules += ['yaml','open3d','scipy','pytorch_lightning']
    elif model_name=='slam':
        required += [root/'vggt_slam/solver.py',root/'third_party/vggt/vggt/models/vggt.py',root/'third_party/salad/salad/models_salad/backbones/dinov2.py',Path(c['torch_home'])/'hub/checkpoints/dino_salad.ckpt',Path(c['torch_home'])/'hub/facebookresearch_dinov2_main/hubconf.py']
        modules += ['gtsam','open3d','scipy','viser']
    for path in required:
        if not path.exists(): errors.append(f'missing required path: {path}')
    for module in modules:
        if importlib.util.find_spec(module) is None: errors.append(f'missing dependency: {module}')
    if model_name=='slam' and importlib.util.find_spec('gtsam'):
        import gtsam
        for attr in ('SL4','PriorFactorSL4','BetweenFactorSL4'):
            if not hasattr(gtsam,attr): errors.append(f'gtsam lacks {attr}; use monst3r interpreter')
    if model_name=='long' and Path(c['dependency_path']).exists():
        from importlib.machinery import PathFinder
        for module in ('pypose','numba','llvmlite','faiss'):
            if PathFinder.find_spec(module,[c['dependency_path']]) is None and importlib.util.find_spec(module) is None: errors.append(f'missing isolated Long dependency: {module}')
    return {'ready':not errors,'errors':errors,'diagnostics':{'model':model_name,'python':sys.executable,'resolved_config':c,'required_paths':[str(p) for p in required],'allocation':'none; no CUDA calls or model constructors'}}


def create_backend(model_name, config, device='cuda'):
    from .runtime import NativeBackend
    return NativeBackend(model_name, resolve_config(model_name,config), device)

__all__=['Prediction','create_backend','doctor_backend']
