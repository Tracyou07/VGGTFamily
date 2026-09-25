"""Tiny, checkpoint-free H20 validation of v8 overlap attention."""
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from tests.ours_v8.test_attention import layout
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.v8.attention import build_correspondences, global_step
from vggt.v6.attention import project_qkv


def flags():
    return dict(flash=torch.backends.cuda.flash_sdp_enabled(),
                memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
                math=torch.backends.cuda.math_sdp_enabled(),
                cudnn=torch.backends.cuda.cudnn_sdp_enabled())


def dense_oracle(block, states, positions, metadata):
    """Explicit metadata mask; no production indexing or pair table."""
    qkv = []
    for state, pos in zip(states, positions):
        x = block.norm1(state)
        a = block.attn
        packed = F.linear(x, a.qkv.weight, a.qkv.bias)
        q, k, v = packed.reshape(1, x.shape[1], 3, a.num_heads,
                                 a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        q, k = a.rope(q, pos), a.rope(k, pos)
        qkv.append((q, k, v))
    q, k, v = (torch.cat([part[j] for part in qkv], dim=2) for j in range(3))
    mask = torch.tensor([
        [src[0] == dst[0] or (src[2] == dst[2] == "patch" and
          src[1] == dst[1] and src[3] == dst[3] and abs(src[0]-dst[0]) == 1)
         for dst in metadata] for src in metadata],
        dtype=torch.bool, device=states[0].device)
    operand = torch.get_autocast_dtype("cuda")
    query = q.to(operand).float()
    key = k.to(operand).float()
    value = v.to(operand).float()
    score = (query @ key.transpose(-2, -1)) * (block.attn.head_dim ** -0.5)
    weights = score.masked_fill(~mask[None, None], -torch.inf).softmax(-1)
    attention = (weights @ value).to(v.dtype)
    attention = attention.transpose(1, 2).reshape(1, len(metadata), -1)
    post = block.attn.proj_drop(block.attn.proj(attention))
    split = post.split([s.shape[1] for s in states], dim=1)
    result = []
    for state, projected in zip(states, split):
        x = state + block.ls1(projected)
        result.append(x + block.ls2(block.mlp(block.norm2(x))))
    return result


def compare(left, right, atol, rtol):
    diff = torch.cat([(a-b).abs().flatten() for a,b in zip(left,right)])
    limits = torch.cat([(atol+rtol*b.abs()).flatten() for b in right])
    return dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                passed=bool((diff<=limits).all()), finite=bool(all(torch.isfinite(x).all() for x in left)),
                atol=atol, rtol=rtol)


def profiled(label, fn):
    torch.cuda.synchronize()
    baseline=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True) as prof:
        output = fn()
        torch.cuda.synchronize()
    rows=[]
    for event in prof.key_averages():
        name=event.key.lower()
        if ((event.key.startswith("aten::") and any(word in name for word in
             ("flash", "cudnn", "scaled_dot", "softmax", "bmm", "index",
              "copy", "gather", "scatter"))) or
             "cudnn_generated" in name or "flash_fwd" in name):
            rows.append(dict(name=event.key, count=event.count,
                             cpu_us=event.self_cpu_time_total,
                             device_us=getattr(event,"self_device_time_total",0)))
    return output, dict(label=label, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        baseline_allocated_bytes=baseline,
                        incremental_peak_bytes=torch.cuda.max_memory_allocated()-baseline,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        profiler_ops=rows)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(87)
    torch.set_grad_enabled(False)
    device=torch.device("cuda:0")
    ids=tuple(f"real-{i+100}" for i in range(5))
    windows=[(0,3),(2,5)]
    states,positions,metadata,count=layout(windows,ids,torch.float32)
    states=[x.to(device) for x in states]
    positions=[x.to(device) for x in positions]
    block=Block(16,2,qk_norm=True,init_values=0.1,
                rope=RotaryPositionEmbedding2D()).to(device).eval()
    mapping=build_correspondences(ids,windows,[ids[a:b] for a,b in windows],
                                   [(2,2)]*2,count,1,2,device)
    before=flags()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        qkv=project_qkv(block.attn,block.norm1(states[0]),positions[0])
        qkv_layout=[dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride())) for x in qkv]
        reference=dense_oracle(block,states,positions,metadata)
        local,local_profile=profiled("native_local",lambda:[
            block(x,pos=p) for x,p in zip(states,positions)])
        correspondence,exchange_profile=profiled("v8_correspondence",lambda:
            global_step(block,states,positions,mapping,"overlap_correspondence",2))
        native,native_profile=profiled("v8_native_sdpa",lambda:
            global_step(block,states,positions,mapping,"overlap_correspondence",2,
                        attention_path="native_sdpa"))
        cached=global_step(block,states,positions,mapping,"overlap_correspondence",2,
                           cache_local_kv_dtype=True)
        independent=build_correspondences(ids,windows,[ids[a:b] for a,b in windows],
                                           [(2,2)]*2,count,1,2,device,"independent")
        independent_output=global_step(block,states,positions,independent,"independent",2)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            forced_flags=flags()
            flash_local=[block(x,pos=p) for x,p in zip(states,positions)]
        after_normal=flags()
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                raise ValueError("intentional scope exit")
        except ValueError:
            pass
        after_exception=flags()
    result=dict(gpu_name=torch.cuda.get_device_name(device),
                gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
                torch=torch.__version__,cuda=torch.version.cuda,
                dtype="torch.bfloat16 autocast",query_chunk_size=2,
                shapes=dict(states=[list(x.shape) for x in states],
                            positions=[list(x.shape) for x in positions],
                            qkv=qkv_layout),
                correspondence_pairs=mapping["pair_count"],
                tolerances=dict(bf16_atol=0.03,bf16_rtol=0.03),
                reference_comparison=compare(correspondence,reference,0.03,0.03),
                native_reference_comparison=compare(native,reference,0.03,0.03),
                native_explicit_comparison=compare(native,correspondence,0.03,0.03),
                cache_on_off=compare(cached,correspondence,0.0,0.0),
                independent_native=compare(independent_output,local,0.0,0.0),
                flash_local=compare(flash_local,local,0.03,0.03),
                sdpa_scope=dict(before=before,inside=forced_flags,
                                after_normal=after_normal,after_exception=after_exception,
                                restored=before==after_normal==after_exception),
                profiles=[local_profile,exchange_profile,native_profile])
    result["passed"]=all((result["reference_comparison"]["passed"],
                          result["cache_on_off"]["passed"],
                          result["native_reference_comparison"]["passed"],
                          result["native_explicit_comparison"]["passed"],
                          result["independent_native"]["passed"],
                          result["sdpa_scope"]["restored"]))
    output=Path(os.environ["V8_GPU_RESULTS"])
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"],
                          reference=result["reference_comparison"],
                          cache=result["cache_on_off"],
                          scope=result["sdpa_scope"],
                          kernel_ops=[p["profiler_ops"] for p in result["profiles"]]),indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
