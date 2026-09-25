"""Small synthetic CUDA diagnostic. Never loads checkpoint or scene images."""
import argparse,fcntl,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from experiments.ours_v5.runtime import preflight
from vggt.layers.attention import Attention
from vggt.v5.attention import project_qkv
lock=open('/tmp/ours_v5_gpu_'+args.gpu+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=preflight(args.gpu,args.output.parent)
if args.output.exists():raise FileExistsError(args.output)
torch.manual_seed(17)
# Same frame/token/head dimensions as the failed 30-frame gate, synthetic values.
a=Attention(1024,16,qk_norm=True).eval().requires_grad_(False).cuda()
x=torch.randn(1,30*1041,1024,device='cuda');camera=torch.arange(0,x.shape[1],1041,device='cuda')
other=torch.arange(x.shape[1],device='cuda');other=other[other%1041!=0]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    q,k,v=project_qkv(a,x,None)
    def observe(fn):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:y=fn()
        torch.cuda.synchronize()
        return y,[event.key for event in prof.key_averages() if 'scaled_dot_product' in event.key]
    full,full_ops=observe(lambda:F.scaled_dot_product_attention(q,k,v))
    cam,cam_ops=observe(lambda:F.scaled_dot_product_attention(q[:,:,camera],k,v,scale=a.scale))
    local,local_ops=observe(lambda:F.scaled_dot_product_attention(q[:,:,other],k,v,scale=a.scale))
    assembled=torch.empty_like(full);assembled[:,:,camera]=cam;assembled[:,:,other]=local
    diff=(full.float()-assembled.float()).abs()
    projected_full=a.proj(full.transpose(1,2).reshape_as(x))
    projected_split=a.proj(assembled.transpose(1,2).reshape_as(x))
    pdiff=(projected_full.float()-projected_split.float()).abs()
    result=dict(gpu=info['gpu_uuid'],shape=list(q.shape),dtype=str(q.dtype),full_ops=full_ops,camera_ops=cam_ops,local_ops=local_ops,
                attention_max_abs=diff.max().item(),attention_mean_abs=diff.mean().item(),attention_changed_elements=int((diff!=0).sum()),
                projected_max_abs=pdiff.max().item(),projected_mean_abs=pdiff.mean().item(),checkpoint_loaded=False)
with args.output.open('x') as out:json.dump(result,out,indent=2)
print(json.dumps(result,indent=2))
