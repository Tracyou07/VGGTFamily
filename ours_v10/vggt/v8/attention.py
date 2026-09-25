"""Exact one-to-one overlap patch exchange in the global attention block.

Each corresponding Query sees its complete local K/V and exactly one remote
K/V. The two branches share a single, stable softmax denominator. No dense
scene mask or per-Query copy of local K/V is materialized.
"""
from collections import defaultdict
import torch
from torch.nn import functional as F

from vggt.v6.attention import _attend, project_qkv

MODES = ("independent", "overlap_correspondence")
ATTENTION_PATHS = ("explicit", "native_sdpa")


def build_correspondences(scene_frame_ids, windows, window_frame_ids, patch_grids,
                          tokens_per_frame, camera_count, register_count, device,
                          mode="overlap_correspondence"):
    """Return indexed patches paired by original frame ID and grid position.

    A frame may occur in at most two adjacent windows. A pair is represented
    in both directions, but the two source token states remain independent.
    """
    if mode not in MODES:
        raise ValueError("unknown v8 mode")
    if not windows or len(windows) != len(window_frame_ids) or len(windows) != len(patch_grids):
        raise ValueError("window, frame-ID and patch-grid lists must agree")
    scene_frame_ids = tuple(scene_frame_ids)
    if len(set(scene_frame_ids)) != len(scene_frame_ids):
        raise ValueError("original frame IDs must be unique")
    if camera_count < 1 or register_count < 0:
        raise ValueError("invalid special token layout")
    patch_start = camera_count + register_count
    if tokens_per_frame <= patch_start:
        raise ValueError("invalid patch token layout")
    grid = tuple(patch_grids[0])
    if len(grid) != 2 or min(grid) < 1 or grid[0] * grid[1] != tokens_per_frame - patch_start:
        raise ValueError("patch grid does not match token layout")
    if any(tuple(other) != grid for other in patch_grids):
        raise ValueError("patch grids differ between windows")

    owners = defaultdict(list)
    for window, ((lo, hi), ids) in enumerate(zip(windows, window_frame_ids)):
        ids = tuple(ids)
        if not 0 <= lo < hi <= len(scene_frame_ids) or ids != scene_frame_ids[lo:hi]:
            raise ValueError("window frame IDs do not match original scene IDs")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate frame ID inside a window")
        for local_frame, frame_id in enumerate(ids):
            owners[frame_id].append((window, local_frame))

    pairs = [defaultdict(lambda: ([], [])) for _ in windows]
    if mode == "overlap_correspondence":
        for frame_id, frame_owners in owners.items():
            if len(frame_owners) > 2:
                raise ValueError(f"multiple remote correspondences for frame {frame_id}")
            if len(frame_owners) == 2:
                (a, fa), (b, fb) = frame_owners
                if abs(a - b) != 1:
                    raise ValueError(f"nonadjacent overlap for frame {frame_id}")
                for source, fs, remote, fr in ((a, fa, b, fb), (b, fb, a, fa)):
                    local_idx, remote_idx = pairs[source][remote]
                    for patch in range(grid[0] * grid[1]):
                        local_idx.append(fs * tokens_per_frame + patch_start + patch)
                        remote_idx.append(fr * tokens_per_frame + patch_start + patch)

    groups = []
    local_queries = []
    pair_count = 0
    for window, (lo, hi) in enumerate(windows):
        count = (hi - lo) * tokens_per_frame
        selected = torch.zeros(count, dtype=torch.bool, device=device)
        window_groups = {}
        for remote, (local_idx, remote_idx) in sorted(pairs[window].items()):
            q = torch.tensor(local_idx, device=device, dtype=torch.long)
            r = torch.tensor(remote_idx, device=device, dtype=torch.long)
            if bool(selected[q].any()):
                raise ValueError("multiple remote correspondences for one Query")
            selected[q] = True
            window_groups[remote] = (q, r)
            pair_count += len(local_idx)
        groups.append(window_groups)
        local_queries.append(torch.arange(count, device=device)[~selected])
    return dict(groups=groups, local_queries=local_queries,
                pair_count=pair_count, patch_grid=grid,
                tokens_per_frame=tokens_per_frame,
                interpretation="one remote K/V per overlapping patch Query; directed count")


def paired_softmax_attention(q, local_k, local_v, remote_k, remote_v,
                             query_chunk_size=16, local_accumulated=None):
    """Stable joint softmax over [complete local K/V, one remote K/V].

    Local score storage is bounded by [B,H,query_chunk_size,local_tokens].
    In autocast, operands are first quantized to its dtype, then scores,
    exponentials and numerator are accumulated in FP32 like fused attention.
    """
    if not isinstance(query_chunk_size, int) or isinstance(query_chunk_size, bool) or query_chunk_size < 1:
        raise ValueError("query chunk size must be a positive integer")
    if q.ndim != 4 or local_k.ndim != 4 or local_v.ndim != 4 or remote_k.shape != q.shape or remote_v.shape != q.shape:
        raise ValueError("expected Q/K/V and one remote K/V for every Query")
    if q.shape[:2] != local_k.shape[:2] or local_k.shape != local_v.shape or q.shape[-1] != local_k.shape[-1]:
        raise ValueError("incompatible attention dimensions")
    device_type = q.device.type
    autocast_on = torch.is_autocast_enabled(device_type)
    operand_dtype = torch.get_autocast_dtype(device_type) if autocast_on else q.dtype
    accumulation_dtype = torch.float64 if operand_dtype == torch.float64 else torch.float32
    outputs = []
    with torch.autocast(device_type=device_type, enabled=False):
        if local_accumulated is None:
            k = local_k.to(operand_dtype).to(accumulation_dtype)
            v = local_v.to(operand_dtype).to(accumulation_dtype)
        else:
            k, v = local_accumulated
            if (k.shape != local_k.shape or v.shape != local_v.shape or
                    k.dtype != accumulation_dtype or v.dtype != accumulation_dtype or
                    k.device != local_k.device or v.device != local_v.device):
                raise ValueError("invalid layer-local K/V conversion cache")
        scale = q.shape[-1] ** -0.5
        for start in range(0, q.shape[-2], query_chunk_size):
            stop = start + query_chunk_size
            query = q[:, :, start:stop].to(operand_dtype).to(accumulation_dtype)
            rk = remote_k[:, :, start:stop].to(operand_dtype).to(accumulation_dtype)
            rv = remote_v[:, :, start:stop].to(operand_dtype).to(accumulation_dtype)
            local_scores = (query @ k.transpose(-2, -1)) * scale
            remote_scores = (query * rk).sum(dim=-1, keepdim=True) * scale
            maximum = torch.maximum(local_scores.max(dim=-1, keepdim=True).values,
                                    remote_scores)
            local_scores.sub_(maximum).exp_()
            remote_weights = (remote_scores - maximum).exp()
            denominator = local_scores.sum(dim=-1, keepdim=True) + remote_weights
            numerator = local_scores @ v + remote_weights * rv
            outputs.append((numerator / denominator).to(local_v.dtype))
    return torch.cat(outputs, dim=-2)


def prepare_native_kv(local_k, local_v, capacity):
    """Copy local K/V once into a reusable bounded native-SDPA buffer."""
    if (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1 or
            local_k.ndim != 4 or local_k.shape != local_v.shape):
        raise ValueError("invalid native K/V buffer")
    tokens = local_k.shape[-2]
    shape = (*local_k.shape[:-2], tokens + capacity, local_k.shape[-1])
    keys = local_k.new_empty(shape); values = local_v.new_empty(shape)
    keys[..., :tokens, :].copy_(local_k); values[..., :tokens, :].copy_(local_v)
    return dict(keys=keys, values=values, local_tokens=tokens, capacity=capacity)


def native_sdpa_attention(q, buffer, remote_k, remote_v, remote_enabled, scale):
    """Native joint SDPA over all local K/V and one diagonal remote K/V."""
    keys, values = buffer["keys"], buffer["values"]
    local_tokens, capacity = buffer["local_tokens"], buffer["capacity"]
    queries = q.shape[-2]
    if (q.ndim != 4 or remote_k.shape != q.shape or remote_v.shape != q.shape or
            queries > capacity or keys.shape[-2] != local_tokens + capacity or
            values.shape != keys.shape or q.shape[:2] != keys.shape[:2] or
            q.shape[-1] != keys.shape[-1]):
        raise ValueError("invalid native SDPA correspondence tensors")
    keys[..., local_tokens:local_tokens + queries, :].copy_(remote_k)
    values[..., local_tokens:local_tokens + queries, :].copy_(remote_v)
    allowed = torch.zeros((queries, local_tokens + capacity), dtype=torch.bool,
                          device=q.device)
    allowed[:, :local_tokens] = True
    if remote_enabled:
        diagonal = torch.arange(queries, device=q.device)
        allowed[diagonal, local_tokens + diagonal] = True
    output = F.scaled_dot_product_attention(
        q, keys, values, attn_mask=allowed, dropout_p=0.0, scale=scale)
    return output, allowed.numel() * allowed.element_size()


def global_step(block, states, positions, correspondence, mode,
                query_chunk_size=16, order=None, memory=None,
                cache_local_kv_dtype=False, attention_path="explicit"):
    """Build every window's Q/K/V before updating any window in this layer."""
    if mode not in MODES or block.training:
        raise ValueError("v8 supports only eval independent or overlap_correspondence")
    if not states or len(states) != len(positions) or len(states) != len(correspondence["groups"]):
        raise ValueError("mismatched window states")
    if not isinstance(query_chunk_size, int) or isinstance(query_chunk_size, bool) or query_chunk_size < 1:
        raise ValueError("query chunk size must be a positive integer")
    if attention_path not in ATTENTION_PATHS:
        raise ValueError("unknown correspondence attention path")
    indices = list(range(len(states))) if order is None else list(order)
    if sorted(indices) != list(range(len(states))):
        raise ValueError("window order must be a permutation")
    result = [None] * len(states)
    if mode == "independent" or correspondence["pair_count"] == 0:
        for window in indices:
            result[window] = block(states[window], pos=positions[window])
        return result

    projected = [project_qkv(block.attn, block.norm1(x), pos)
                 for x, pos in zip(states, positions)]
    if memory is not None:
        memory["projected_qkv_bytes"] = max(memory.get("projected_qkv_bytes", 0),
            sum(t.numel() * t.element_size() for triplet in projected for t in triplet))
    attention = block.attn
    for window in indices:
        q, k, v = projected[window]
        output = torch.empty(q.shape, device=q.device, dtype=v.dtype)
        local = correspondence["local_queries"][window]
        if local.numel():
            output[:, :, local] = _attend(attention, q[:, :, local], k, v)
        local_accumulated = None
        native_buffer = None
        if attention_path == "native_sdpa" and correspondence["groups"][window]:
            native_buffer = prepare_native_kv(k, v, query_chunk_size)
            if memory is not None:
                size = sum(x.numel() * x.element_size()
                           for x in (native_buffer["keys"], native_buffer["values"]))
                memory["native_kv_buffer_bytes"] = max(memory.get("native_kv_buffer_bytes", 0), size)
        elif cache_local_kv_dtype and correspondence["groups"][window]:
            device_type = q.device.type
            operand_dtype = (torch.get_autocast_dtype(device_type)
                             if torch.is_autocast_enabled(device_type) else q.dtype)
            accumulation_dtype = (torch.float64 if operand_dtype == torch.float64
                                  else torch.float32)
            local_accumulated = (k.to(operand_dtype).to(accumulation_dtype),
                                 v.to(operand_dtype).to(accumulation_dtype))
        for remote_window, (query_indices, remote_indices) in correspondence["groups"][window].items():
            remote_k, remote_v = projected[remote_window][1:]
            for start in range(0, query_indices.numel(), query_chunk_size):
                q_idx = query_indices[start:start + query_chunk_size]
                r_idx = remote_indices[start:start + query_chunk_size]
                if attention_path == "native_sdpa":
                    value, mask_bytes = native_sdpa_attention(
                        q[:, :, q_idx], native_buffer,
                        remote_k[:, :, r_idx], remote_v[:, :, r_idx], True,
                        attention.scale)
                    output[:, :, q_idx] = value
                    if memory is not None:
                        memory["native_mask_bytes"] = max(memory.get("native_mask_bytes", 0), mask_bytes)
                else:
                    output[:, :, q_idx] = paired_softmax_attention(
                        q[:, :, q_idx], k, v,
                        remote_k[:, :, r_idx], remote_v[:, :, r_idx],
                        query_chunk_size=query_chunk_size,
                        local_accumulated=local_accumulated)
        del local_accumulated, native_buffer
        y = attention.proj_drop(attention.proj(output.transpose(1, 2).reshape_as(states[window])))
        x = states[window] + block.ls1(y)
        result[window] = x + block.ls2(block.mlp(block.norm2(x)))
    return result
