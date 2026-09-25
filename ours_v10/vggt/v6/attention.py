"""Allowed-edge attention for independent camera/register window instances."""
import hashlib
import math
import torch
from torch.nn import functional as F

MODES = ("independent", "camera_exchange", "camera_register_exchange")


def layout(aggregator):
    camera = int(aggregator.camera_token.shape[2])
    register = int(aggregator.register_token.shape[2])
    if camera < 1 or register < 1 or camera + register != int(aggregator.patch_start_idx):
        raise ValueError("invalid camera/register token layout")
    return camera, register


def special_indices(length, tokens_per_frame, count, device):
    if count < 1 or count > tokens_per_frame or length % tokens_per_frame:
        raise ValueError("invalid special-token layout")
    frame = torch.arange(length // tokens_per_frame, device=device)
    offset = torch.arange(count, device=device)
    return (frame[:, None] * tokens_per_frame + offset).reshape(-1)


def project_qkv(attention, x, pos):
    b, n, _ = x.shape
    q, k, v = attention.qkv(x).reshape(
        b, n, 3, attention.num_heads, attention.head_dim
    ).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attention.q_norm(q), attention.k_norm(k)
    if attention.rope is not None:
        q = attention.rope(q, pos)
        k = attention.rope(k, pos)
    return q, k, v


def communication_bank(block, old_state, pos, tokens_per_frame, count):
    indices = special_indices(old_state.shape[1], tokens_per_frame, count, old_state.device)
    selected = block.norm1(old_state[:, indices])
    selected_pos = None if pos is None else pos[:, indices]
    _, k, v = project_qkv(block.attn, selected, selected_pos)
    return k, v


def _attend(attention, q, k, v):
    if attention.fused_attn:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=attention.scale)
    return ((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1) @ v


def exchange_attention(attention, x, pos, tokens_per_frame, bank, window_index,
                       query_count, query_chunk_size=64, memory=None):
    if attention.training:
        raise ValueError("v6 attention is inference-only")
    if x.shape[0] != 1 or x.shape[1] % tokens_per_frame:
        raise ValueError("one complete window instance required")
    if query_chunk_size < 1 or not 0 <= window_index < len(bank):
        raise ValueError("invalid query chunk or window index")
    if len(bank) == 1:
        return attention(x, pos=pos)
    q, k, v = project_qkv(attention, x, pos)
    special = special_indices(x.shape[1], tokens_per_frame, query_count, x.device)
    local_only = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
    local_only[special] = False
    other = torch.arange(x.shape[1], device=x.device)[local_only]
    remote = [pair for j, pair in enumerate(bank) if j != window_index]
    keys = torch.cat([k] + [pair[0] for pair in remote], dim=2)
    values = torch.cat([v] + [pair[1] for pair in remote], dim=2)
    if memory is not None:
        memory["temporary_qkv_bytes"] = max(memory.get("temporary_qkv_bytes", 0),
            sum(t.numel() * t.element_size() for t in (q, k, v)))
        memory["concatenated_kv_bytes"] = max(memory.get("concatenated_kv_bytes", 0),
            sum(t.numel() * t.element_size() for t in (keys, values)))
    output = None
    for start in range(0, len(special), query_chunk_size):
        indices = special[start:start + query_chunk_size]
        y = _attend(attention, q[:, :, indices], keys, values)
        if output is None:
            output = torch.empty(q.shape, device=q.device, dtype=y.dtype)
        output[:, :, indices] = y
    if len(other):
        output[:, :, other] = _attend(attention, q[:, :, other], k, v)
    output = output.transpose(1, 2).reshape_as(x)
    return attention.proj_drop(attention.proj(output))


def _checksum(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sample_indices(frames, tokens_per_frame, offset, max_frames, device):
    frame_ids = sorted(set(round(i * (frames - 1) / max(1, min(max_frames, frames) - 1))
                           for i in range(min(max_frames, frames))))
    return torch.tensor([f * tokens_per_frame + offset for f in frame_ids], device=device)


def _diagnose_window(block, state, pos, bank, index, mode, layer, tokens_per_frame,
                     camera_count, register_count, bank_bytes, sample_frames=4):
    """Opt-in read-only probe; production SDPA output is computed independently."""
    frames = state.shape[1] // tokens_per_frame
    camera_idx = special_indices(state.shape[1], tokens_per_frame, camera_count, state.device)
    reg_idx = special_indices(state.shape[1], tokens_per_frame, camera_count + register_count, state.device)
    reg_idx = reg_idx.reshape(frames, camera_count + register_count)[:, camera_count:].reshape(-1)
    normalized = block.norm1(state)
    q, k, v = project_qkv(block.attn, normalized, pos)
    remote_pairs = [] if bank is None else [pair for j, pair in enumerate(bank) if j != index]
    rk = torch.cat([p[0] for p in remote_pairs], dim=2) if remote_pairs else None
    rv = torch.cat([p[1] for p in remote_pairs], dim=2) if remote_pairs else None
    local_n = int(k.shape[2]); remote_n = 0 if rk is None else int(rk.shape[2])
    query_limit = camera_count if mode == 'camera_exchange' else camera_count + register_count if mode == 'camera_register_exchange' else 0
    per_head = [dict(head=h) for h in range(block.attn.num_heads)]
    type_rows = {}
    for kind, offsets in [('camera',range(camera_count)),('register',range(camera_count,camera_count+register_count))]:
        indices = torch.cat([_sample_indices(frames,tokens_per_frame,o,sample_frames,state.device) for o in offsets])
        enabled = offsets.start < query_limit and remote_n > 0
        if enabled:
            qs=q[:,:,indices].float()
            all_k=torch.cat((k,rk),dim=2).float()
            logits=(qs*block.attn.scale)@all_k.transpose(-1,-2)
            probs=logits.softmax(-1)
            masses=probs[...,local_n:].sum(-1)
            entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1)/math.log(local_n+remote_n)
            remote_out=probs@torch.cat((v,rv),dim=2).float()
            local_out=((qs*block.attn.scale)@k.float().transpose(-1,-2)).softmax(-1)@v.float()
            delta=(remote_out-local_out).norm(dim=-1)
        else:
            masses=torch.zeros((1,block.attn.num_heads,len(indices)),device=state.device)
            entropy=torch.zeros_like(masses)
            delta=torch.zeros_like(masses)
        type_rows[kind]=dict(sampled_queries=len(indices),remote_attention_mass=float(masses.mean()),
                             normalized_entropy=float(entropy.mean()),query_delta_norm=float(delta.mean()))
        for h in range(block.attn.num_heads):
            per_head[h][kind]=dict(remote_attention_mass=float(masses[0,h].mean()),
                                   normalized_entropy=float(entropy[0,h].mean()),
                                   query_delta_norm=float(delta[0,h].mean()))
    local_knorm=k.float().norm(dim=-1).mean((0,2)).cpu().tolist()
    local_vnorm=v.float().norm(dim=-1).mean((0,2)).cpu().tolist()
    remote_knorm=[0.]*block.attn.num_heads if rk is None else rk.float().norm(dim=-1).mean((0,2)).cpu().tolist()
    remote_vnorm=[0.]*block.attn.num_heads if rv is None else rv.float().norm(dim=-1).mean((0,2)).cpu().tolist()
    for h,row in enumerate(per_head):
        row.update(local_k_norm=local_knorm[h],local_v_norm=local_vnorm[h],
                   remote_k_norm=remote_knorm[h],remote_v_norm=remote_vnorm[h])
    all_special=normalized[:,special_indices(state.shape[1],tokens_per_frame,camera_count+register_count,state.device)]
    bank_checksum='none' if bank is None else _checksum(torch.cat((bank[index][0].flatten(),bank[index][1].flatten())))
    remote_checksums=[] if bank is None else [_checksum(torch.cat((pair[0].flatten(),pair[1].flatten()))) for j,pair in enumerate(bank) if j!=index]
    weight_camera=camera_count if query_limit else 0
    weight_register=register_count if query_limit>camera_count else 0
    weight_total=max(1,weight_camera+weight_register)
    remote_mass=(weight_camera*type_rows['camera']['remote_attention_mass']+weight_register*type_rows['register']['remote_attention_mass'])/weight_total
    special_delta=(weight_camera*type_rows['camera']['query_delta_norm']+weight_register*type_rows['register']['query_delta_norm'])/weight_total
    return dict(mode=mode,layer=layer,window_index=index,window_count=1 if bank is None else len(bank),
                camera_count=camera_count,register_count=register_count,local_kv_tokens=local_n,
                remote_kv_tokens=remote_n,communication_bank_bytes=bank_bytes,
                remote_attention_mass=remote_mass,local_attention_mass=1-remote_mass,
                special_query_delta_norm=special_delta,
                patch_query_delta_norm=0.,heads=per_head,query_types=type_rows,
                bank_checksum=bank_checksum,remote_bank_checksums=remote_checksums,token_layout=dict(tokens_per_frame=tokens_per_frame,
                    patch_start_idx=camera_count+register_count,
                    camera_indices=[int(camera_idx[0]),int(camera_idx[-1])],
                    register_indices=[int(reg_idx[0]),int(reg_idx[-1])],
                    special_input_checksum=_checksum(all_special),
                    camera_checksum=_checksum(normalized[:,camera_idx]),
                    register_checksum=_checksum(normalized[:,reg_idx])))


def global_step(block, states, positions, tokens_per_frame, mode, camera_count,
                register_count, query_chunk_size=64, order=None, memory=None,
                diagnostics=None, layer=None):
    if mode not in MODES:
        raise ValueError("unknown communication mode")
    if block.training:
        raise ValueError("v6 is inference-only")
    if not states or len(states) != len(positions):
        raise ValueError("empty or mismatched scene")
    if camera_count < 1 or register_count < 0 or camera_count + register_count > tokens_per_frame:
        raise ValueError("invalid special-token counts")
    indices = list(range(len(states))) if order is None else list(order)
    if sorted(indices) != list(range(len(states))):
        raise ValueError("order must be a permutation")
    count = camera_count if mode == "camera_exchange" else camera_count + register_count
    bank = None
    if mode != "independent" and len(states) > 1:
        bank = [communication_bank(block, x, p, tokens_per_frame, count)
                for x, p in zip(states, positions)]
        if memory is not None:
            memory["communication_bank_bytes"] = max(memory.get("communication_bank_bytes", 0),
                sum(t.numel() * t.element_size() for pair in bank for t in pair))
    bank_bytes=0 if bank is None else sum(t.numel()*t.element_size() for pair in bank for t in pair)
    result = [None] * len(states)
    for i in indices:
        x = states[i]
        if bank is None:
            result[i] = block(x, pos=positions[i])
            continue
        y = exchange_attention(block.attn, block.norm1(x), positions[i],
                               tokens_per_frame, bank, i, count, query_chunk_size, memory)
        x = x + block.ls1(y)
        result[i] = x + block.ls2(block.mlp(block.norm2(x)))
    if diagnostics is not None:
        if layer is None:
            raise ValueError('diagnostic layer required')
        for i in range(len(states)):
            row=_diagnose_window(block,states[i],positions[i],bank,i,mode,layer,tokens_per_frame,
                                 camera_count,register_count,bank_bytes)
            row['window_count']=len(states)
            diagnostics.append(row)
    return result
