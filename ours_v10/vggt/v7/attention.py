"""Original-block attention over local tokens plus selected remote camera/patch K/V.

All banks are derived from the same pre-global-block states. No scene-wide
attention score or dense mask is constructed in this production path.
"""
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from vggt.v6.attention import _attend, layout, project_qkv

MODES = ("independent", "camera_exchange", "camera_patch_exchange")


def resolve_query_chunks(query_chunk_size, local_query_chunk_size,
                         cross_query_chunk_size):
    if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query chunk size must be a positive integer")
    if query_chunk_size != 64 and (local_query_chunk_size is not None or
                                   cross_query_chunk_size is not None):
        raise ValueError("legacy query chunk size conflicts with group-specific chunks")
    def resolve(value):
        if value is None:
            return query_chunk_size
        if value == "all":
            return value
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("group query chunk must be a positive integer or 'all'")
        return value
    return resolve(local_query_chunk_size), resolve(cross_query_chunk_size)


def exchange_sdpa_scope(backend):
    """Limit only v7 exchanged queries to a selected SDPA kernel."""
    if backend == "auto":
        return nullcontext()
    if backend == "flash":
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    raise ValueError(f"unknown exchange SDPA backend: {backend}")


def selected_indices(length, tokens_per_frame, camera_count, register_count,
                     patch_positions, mode, device):
    if mode not in MODES or tokens_per_frame < 1 or length % tokens_per_frame:
        raise ValueError("invalid mode or token layout")
    patch_start = camera_count + register_count
    patch_count = tokens_per_frame - patch_start
    if camera_count < 1 or register_count < 0 or patch_count < 1:
        raise ValueError("invalid model token layout")
    positions = tuple(int(i) for i in patch_positions)
    if len(set(positions)) != len(positions) or any(i < 0 or i >= patch_count for i in positions):
        raise ValueError("patch selection is not unique or is outside patch grid")
    if mode == "independent":
        offsets = ()
    elif mode == "camera_exchange":
        offsets = tuple(range(camera_count))
    else:
        offsets = tuple(range(camera_count)) + tuple(patch_start + i for i in positions)
    frame_count = length // tokens_per_frame
    return torch.tensor([frame * tokens_per_frame + offset
                         for frame in range(frame_count) for offset in offsets],
                        device=device, dtype=torch.long)


def communication_bank(block, old_state, pos, tokens_per_frame,
                       camera_count, register_count, patch_positions, mode):
    idx = selected_indices(old_state.shape[1], tokens_per_frame,
                           camera_count, register_count, patch_positions,
                           mode, old_state.device)
    selected = block.norm1(old_state[:, idx])
    selected_pos = None if pos is None else pos[:, idx]
    _, k, v = project_qkv(block.attn, selected, selected_pos)
    return k, v


def exchange_attention(attention, x, pos, tokens_per_frame, bank, window_index,
                       camera_count, register_count, patch_positions, mode,
                       query_chunk_size=64, memory=None, exchange_sdpa_backend="auto",
                       local_query_chunk_size=None, cross_query_chunk_size=None):
    if attention.training:
        raise ValueError("v7 attention is inference-only")
    if x.shape[0] != 1 or x.shape[1] % tokens_per_frame:
        raise ValueError("one complete window instance required")
    if not 0 <= window_index < len(bank):
        raise ValueError("invalid query chunk or window index")
    local_chunk, cross_chunk = resolve_query_chunks(
        query_chunk_size, local_query_chunk_size, cross_query_chunk_size)
    sdpa_scope = exchange_sdpa_scope(exchange_sdpa_backend)
    q, k, v = project_qkv(attention, x, pos)
    cross = selected_indices(x.shape[1], tokens_per_frame, camera_count,
                             register_count, patch_positions, mode, x.device)
    local_mask = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
    local_mask[cross] = False
    local = torch.arange(x.shape[1], device=x.device)[local_mask]
    remote = [pair for j, pair in enumerate(bank) if j != window_index]
    keys = torch.cat([k] + [pair[0] for pair in remote], dim=2)
    values = torch.cat([v] + [pair[1] for pair in remote], dim=2)
    if memory is not None:
        memory["temporary_qkv_bytes"] = max(memory.get("temporary_qkv_bytes", 0),
            sum(t.numel() * t.element_size() for t in (q, k, v)))
        memory["concatenated_kv_bytes"] = max(memory.get("concatenated_kv_bytes", 0),
            sum(t.numel() * t.element_size() for t in (keys, values)))
    output = None
    with sdpa_scope:
        cross_step = max(1, len(cross)) if cross_chunk == "all" else cross_chunk
        local_step = max(1, len(local)) if local_chunk == "all" else local_chunk
        for start in range(0, len(cross), cross_step):
            indices = cross[start:start + cross_step]
            y = _attend(attention, q[:, :, indices], keys, values)
            if output is None:
                output = torch.empty(q.shape, device=q.device, dtype=y.dtype)
            output[:, :, indices] = y
        for start in range(0, len(local), local_step):
            indices = local[start:start + local_step]
            y = _attend(attention, q[:, :, indices], k, v)
            if output is None:
                output = torch.empty(q.shape, device=q.device, dtype=y.dtype)
            output[:, :, indices] = y
    output = output.transpose(1, 2).reshape_as(x)
    return attention.proj_drop(attention.proj(output))


def global_step(block, states, positions, tokens_per_frame, mode, camera_count,
                register_count, patch_positions=(), query_chunk_size=64,
                order=None, memory=None, exchange_sdpa_backend="auto",
                local_query_chunk_size=None, cross_query_chunk_size=None):
    if mode not in MODES:
        raise ValueError("unknown communication mode")
    if block.training:
        raise ValueError("v7 is inference-only")
    if not states or len(states) != len(positions):
        raise ValueError("empty or mismatched scene")
    resolve_query_chunks(query_chunk_size, local_query_chunk_size,
                         cross_query_chunk_size)
    indices = list(range(len(states))) if order is None else list(order)
    if sorted(indices) != list(range(len(states))):
        raise ValueError("order must be a permutation")
    # Validate selection even in independent mode, then build every bank before
    # mutating any window. The bank contains only that source window's tokens.
    patch_positions = tuple(patch_positions)
    for state in states:
        selected_indices(state.shape[1], tokens_per_frame, camera_count,
                         register_count, patch_positions, mode, state.device)
    bank = None
    if mode != "independent" and len(states) > 1:
        bank = [communication_bank(block, x, p, tokens_per_frame, camera_count,
                                   register_count, patch_positions, mode)
                for x, p in zip(states, positions)]
        if memory is not None:
            memory["communication_bank_bytes"] = max(memory.get("communication_bank_bytes", 0),
                sum(t.numel() * t.element_size() for pair in bank for t in pair))
    result = [None] * len(states)
    for i in indices:
        x = states[i]
        if bank is None:
            result[i] = block(x, pos=positions[i])
            continue
        y = exchange_attention(block.attn, block.norm1(x), positions[i],
                               tokens_per_frame, bank, i, camera_count,
                               register_count, patch_positions, mode,
                               query_chunk_size, memory,
                               exchange_sdpa_backend=exchange_sdpa_backend,
                               local_query_chunk_size=local_query_chunk_size,
                               cross_query_chunk_size=cross_query_chunk_size)
        x = x + block.ls1(y)
        result[i] = x + block.ls2(block.mlp(block.norm2(x)))
    return result
