"""Replay one cached global block only, never a full model/sequence forward."""
import argparse,fcntl,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import torch
from safetensors import safe_open
from torch.nn import functional as F
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D,PositionGetter
from vggt.v5.attention import global_step,project_qkv,exchange_attention
from experiments.ours_v5.runtime import preflight
lock=open('/tmp/ours_v5_gpu_'+args.gpu+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
info=preflight(args.gpu,args.output.parent)
if args.output.exists():raise FileExistsError(args.output)
torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
folder=Path('/data/yjh/output/vggt/ours_v5/20260921T042854Z_v5_single/bf16_reference')
saved=torch.load(folder/'window_0000.pt',mmap=True,map_location='cpu',weights_only=True)
manifest=json.loads((folder/'run_manifest.json').read_text());h,w=manifest['preprocessing']['shape'][-2:]
cache=saved['head_cache_4'];s,p,c2=cache.shape;c=c2//2
x=cache[...,:c].reshape(1,s*p,c).cuda();expected=cache[...,c:].reshape(1,s*p,c).cuda()
spatial=PositionGetter()(s,h//14,w//14,'cuda')+1
pos=torch.cat([torch.zeros(s,5,2,device='cuda',dtype=spatial.dtype),spatial],dim=1).reshape(1,s*p,2)
b=Block(c,16,init_values=.01,qk_norm=True,rope=RotaryPositionEmbedding2D(100)).eval().requires_grad_(False)
with safe_open('/data/yjh/share/pretrained/VGGT-1B/model.safetensors',framework='pt',device='cpu') as f:
    b.load_state_dict({k:f.get_tensor('aggregator.global_blocks.4.'+k) for k in b.state_dict()},strict=True)
b.cuda()
def delta(a,b):
    d=(a.float()-b.float()).abs();return dict(max_abs=d.max().item(),mean_abs=d.mean().item(),changed=int((d!=0).sum()))
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    normal=b(x,pos=pos);split=global_step(b,[x],[pos],p,'camera_exchange')[0]
    z=b.norm1(x);q,k,v=project_qkv(b.attn,z,pos)
    ci=torch.arange(0,s*p,p,device='cuda');ni=torch.arange(s*p,device='cuda');ni=ni[ni%p!=0]
    full=F.scaled_dot_product_attention(q,k,v)
    ca=F.scaled_dot_product_attention(q[:,:,ci],k,v,scale=b.attn.scale)
    no=F.scaled_dot_product_attention(q[:,:,ni],k,v,scale=b.attn.scale)
    output=torch.empty(q.shape,device='cuda',dtype=ca.dtype);output[:,:,ci]=ca;output[:,:,ni]=no
    attn_normal=b.attn(z,pos)
    attn_split=b.attn.proj(output.transpose(1,2).reshape_as(x))
    result=dict(gpu=info['gpu_uuid'],layer=4,shape=list(q.shape),q_dtype=str(q.dtype),v_dtype=str(v.dtype),output_dtype=str(full.dtype),
                original_vs_saved=delta(normal,expected),block_original_vs_split=delta(normal,split),sdpa_whole_vs_split=delta(full,output),attention_original_vs_split=delta(attn_normal,attn_split),
                full_stride=list(full.stride()),scatter_stride=list(output.stride()),full_model_run=False)
torch.cuda.synchronize()
with args.output.open('x') as out:json.dump(result,out,indent=2)
print(json.dumps(result,indent=2))
