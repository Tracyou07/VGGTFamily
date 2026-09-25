"""Global camera-to-camera bank combined with v9 overlap patch correspondence.

The v8 attention path is used unchanged for its two inherited modes. New modes
project every window before any global-block update, retain independent token
instances, and normalize every Query over its entire allowed K/V set once.
"""
from collections import Counter

import torch
from torch.nn import functional as F

from vggt.v6.attention import _attend, project_qkv, special_indices
from vggt.v8.attention import (
    build_correspondences as v8_build_correspondences,
    global_step as v8_global_step,
    native_sdpa_attention, paired_softmax_attention, prepare_native_kv,
)

MODES = ("independent", "camera_only", "overlap_correspondence",
         "camera_global_overlap")
CAMERA_MODES = ("camera_only", "camera_global_overlap")
OVERLAP_MODES = ("overlap_correspondence", "camera_global_overlap")


def build_correspondences(scene_frame_ids, windows, window_frame_ids,
                          patch_grids, tokens_per_frame, camera_count,
                          register_count, device, mode="camera_global_overlap"):
    if mode not in MODES:
        raise ValueError("unknown v10 mode")
    base_mode = "overlap_correspondence" if mode in OVERLAP_MODES else "independent"
    result = v8_build_correspondences(scene_frame_ids, windows, window_frame_ids,
        patch_grids, tokens_per_frame, camera_count, register_count, device,
        base_mode)
    all_ids = [str(frame) for ids in window_frame_ids for frame in ids]
    result["duplicate_window_frame_instances"] = len(all_ids)-len(set(all_ids))
    result["camera_count"] = camera_count
    result["camera_queries"] = []
    result["camera_bank"] = []
    for target,(lo,hi) in enumerate(windows):
        local = set(str(frame) for frame in window_frame_ids[target])
        camera = special_indices((hi-lo)*tokens_per_frame, tokens_per_frame,
                                 camera_count, device)
        result["camera_queries"].append(camera if mode in CAMERA_MODES else camera[:0])
        if mode in CAMERA_MODES:
            selected = torch.ones((hi-lo)*tokens_per_frame,
                                  dtype=torch.bool, device=device)
            selected[camera] = False
            result["local_queries"][target] = result["local_queries"][target][
                selected[result["local_queries"][target]]]
        remote = []
        remote_ids = []
        for source,(start,end) in enumerate(windows):
            if source == target:
                continue
            indices = special_indices((end-start)*tokens_per_frame,
                                      tokens_per_frame, camera_count, device)
            remote.append((source, indices))
            remote_ids.extend(str(frame) for frame in window_frame_ids[source])
        counts = Counter(remote_ids)
        result["camera_bank"].append(dict(
            remote_indices=remote,
            remote_tokens=sum(len(indices) for _,indices in remote),
            remote_duplicate_frame_instances=sum(n-1 for n in counts.values()),
            remote_also_local_frame_instances=sum(frame in local for frame in remote_ids)))
    result["interpretation"] = (
        "all other-window camera instances plus one-to-one adjacent overlap patch"
        if mode == "camera_global_overlap" else
        "all other-window camera instances" if mode == "camera_only" else
        result["interpretation"])
    return result


def _camera_attention(attention, q, k, v, projected, bank, path):
    remote_keys = [projected[source][1][:,:,idx] for source,idx in bank["remote_indices"]]
    remote_values = [projected[source][2][:,:,idx] for source,idx in bank["remote_indices"]]
    keys = torch.cat([k,*remote_keys], dim=2)
    values = torch.cat([v,*remote_values], dim=2)
    if path == "native_sdpa":
        output = F.scaled_dot_product_attention(
            q, keys, values, dropout_p=0.0, scale=attention.scale)
    else:
        output = ((q * attention.scale) @ keys.transpose(-2,-1)).softmax(-1) @ values
    bytes_used = (keys.numel()*keys.element_size() +
                  values.numel()*values.element_size())
    return output, bytes_used


def _global_step_streamed(block, states, positions, correspondence, mode,
                          query_chunk_size, indices, memory,
                          cache_local_kv_dtype, attention_path):
    """Keep the layer barrier while retaining only remotely needed K/V.

    Every source is projected with the original full-window QKV operation.
    Camera and matched-overlap K/V are copied into a compact immutable bank;
    the complete projection is then released. A target recomputes its own
    full-window QKV immediately before its unchanged local attention math.
    No result state is substituted for a same-layer source state.
    """
    requested_by_source = [[] for _ in states]
    for target, groups in enumerate(correspondence["groups"]):
        for source, (_queries, remote_indices) in groups.items():
            requested_by_source[source].append((target, remote_indices))

    camera_bank = [None] * len(states)
    overlap_bank = {}
    attention = block.attn
    for source, (state, pos) in enumerate(zip(states, positions)):
        q, k, v = project_qkv(attention, block.norm1(state), pos)
        if memory is not None:
            size = sum(t.numel() * t.element_size() for t in (q, k, v))
            memory["projected_qkv_bytes"] = max(
                memory.get("projected_qkv_bytes", 0), size)
        camera_indices = correspondence["camera_queries"][source]
        camera_bank[source] = (k[:, :, camera_indices], v[:, :, camera_indices])
        for target, remote_indices in requested_by_source[source]:
            overlap_bank[target, source] = (
                k[:, :, remote_indices], v[:, :, remote_indices])
        del q, k, v

    if memory is not None:
        memory["streamed_bank_bytes"] = sum(
            value.numel() * value.element_size()
            for pair in camera_bank for value in pair
        ) + sum(
            value.numel() * value.element_size()
            for pair in overlap_bank.values() for value in pair
        )

    result = [None] * len(states)
    for window in indices:
        q, k, v = project_qkv(attention, block.norm1(states[window]), positions[window])
        output = torch.empty(q.shape, device=q.device, dtype=v.dtype)
        local = correspondence["local_queries"][window]
        if local.numel():
            output[:, :, local] = _attend(attention, q[:, :, local], k, v)

        camera = correspondence["camera_queries"][window]
        if camera.numel():
            bank = correspondence["camera_bank"][window]
            remote_keys = [camera_bank[source][0]
                           for source, _indices in bank["remote_indices"]]
            remote_values = [camera_bank[source][1]
                             for source, _indices in bank["remote_indices"]]
            camera_keys = torch.cat([k, *remote_keys], dim=2)
            camera_values = torch.cat([v, *remote_values], dim=2)
            if attention_path == "native_sdpa":
                camera_output = F.scaled_dot_product_attention(
                    q[:, :, camera], camera_keys, camera_values,
                    dropout_p=0.0, scale=attention.scale)
            else:
                camera_output = ((q[:, :, camera] * attention.scale)
                                 @ camera_keys.transpose(-2, -1)).softmax(-1) @ camera_values
            output[:, :, camera] = camera_output
            if memory is not None:
                size = sum(t.numel() * t.element_size()
                           for t in (camera_keys, camera_values))
                memory["camera_bank_kv_peak_bytes"] = max(
                    memory.get("camera_bank_kv_peak_bytes", 0), size)
            del camera_keys, camera_values, camera_output

        native_buffer = None
        local_accumulated = None
        if correspondence["groups"][window]:
            if attention_path == "native_sdpa":
                native_buffer = prepare_native_kv(k, v, query_chunk_size)
                if memory is not None:
                    size = sum(t.numel() * t.element_size() for t in
                               (native_buffer["keys"], native_buffer["values"]))
                    memory["native_kv_buffer_bytes"] = max(
                        memory.get("native_kv_buffer_bytes", 0), size)
            elif cache_local_kv_dtype:
                device_type = q.device.type
                operand_dtype = (torch.get_autocast_dtype(device_type)
                    if torch.is_autocast_enabled(device_type) else q.dtype)
                accumulated_dtype = (torch.float64 if operand_dtype == torch.float64
                                     else torch.float32)
                local_accumulated = (k.to(operand_dtype).to(accumulated_dtype),
                                     v.to(operand_dtype).to(accumulated_dtype))
        for source, (query_indices, _remote_indices) in correspondence["groups"][window].items():
            remote_k, remote_v = overlap_bank[window, source]
            for start in range(0, len(query_indices), query_chunk_size):
                stop = start + query_chunk_size
                qidx = query_indices[start:stop]
                rk = remote_k[:, :, start:stop]
                rv = remote_v[:, :, start:stop]
                if attention_path == "native_sdpa":
                    value, mask_bytes = native_sdpa_attention(
                        q[:, :, qidx], native_buffer, rk, rv, True, attention.scale)
                    output[:, :, qidx] = value
                    if memory is not None:
                        memory["native_mask_bytes"] = max(
                            memory.get("native_mask_bytes", 0), mask_bytes)
                else:
                    output[:, :, qidx] = paired_softmax_attention(
                        q[:, :, qidx], k, v, rk, rv,
                        query_chunk_size=query_chunk_size,
                        local_accumulated=local_accumulated)
        del native_buffer, local_accumulated
        y = attention.proj_drop(attention.proj(
            output.transpose(1, 2).reshape_as(states[window])))
        x = states[window] + block.ls1(y)
        result[window] = x + block.ls2(block.mlp(block.norm2(x)))
        del q, k, v, output, y, x
    return result


def global_step(block, states, positions, correspondence, mode,
                query_chunk_size=16, order=None, memory=None,
                cache_local_kv_dtype=False, attention_path="explicit",
                stream_projected_qkv=False):
    if mode not in MODES:
        raise ValueError("unknown v10 mode")
    if stream_projected_qkv and mode not in CAMERA_MODES:
        raise ValueError("streamed QKV is supported only for camera modes")
    if mode not in CAMERA_MODES:
        return v8_global_step(block, states, positions, correspondence, mode,
            query_chunk_size=query_chunk_size, order=order, memory=memory,
            cache_local_kv_dtype=cache_local_kv_dtype, attention_path=attention_path)
    if block.training or not states or len(states) != len(positions):
        raise ValueError("camera communication requires nonempty eval window states")
    if len(states) != len(correspondence["groups"]):
        raise ValueError("mismatched correspondence windows")
    if attention_path not in ("explicit", "native_sdpa") or query_chunk_size < 1:
        raise ValueError("invalid attention path or query chunk")
    indices = list(range(len(states))) if order is None else list(order)
    if sorted(indices) != list(range(len(states))):
        raise ValueError("window order must be a permutation")
    if len(states) == 1 and correspondence["pair_count"] == 0:
        return [block(states[0], pos=positions[0])]

    if stream_projected_qkv:
        return _global_step_streamed(
            block, states, positions, correspondence, mode,
            query_chunk_size, indices, memory, cache_local_kv_dtype,
            attention_path)

    # Barrier: all Q/K/V are immutable inputs for this global layer.
    projected = [project_qkv(block.attn, block.norm1(x), pos)
                 for x,pos in zip(states,positions)]
    if memory is not None:
        memory["projected_qkv_bytes"] = max(memory.get("projected_qkv_bytes",0),
            sum(t.numel()*t.element_size() for triple in projected for t in triple))
    attention = block.attn
    result = [None]*len(states)
    for window in indices:
        q,k,v = projected[window]
        output = torch.empty(q.shape, device=q.device, dtype=v.dtype)
        local = correspondence["local_queries"][window]
        if local.numel():
            output[:,:,local] = _attend(attention, q[:,:,local], k, v)

        camera = correspondence["camera_queries"][window]
        if camera.numel():
            value, bank_bytes = _camera_attention(attention, q[:,:,camera], k, v,
                projected, correspondence["camera_bank"][window], attention_path)
            output[:,:,camera] = value
            if memory is not None:
                memory["camera_bank_kv_peak_bytes"] = max(
                    memory.get("camera_bank_kv_peak_bytes",0), bank_bytes)

        native_buffer = None
        local_accumulated = None
        if correspondence["groups"][window]:
            if attention_path == "native_sdpa":
                native_buffer = prepare_native_kv(k,v,query_chunk_size)
                if memory is not None:
                    size = sum(t.numel()*t.element_size() for t in
                               (native_buffer["keys"],native_buffer["values"]))
                    memory["native_kv_buffer_bytes"] = max(
                        memory.get("native_kv_buffer_bytes",0),size)
            elif cache_local_kv_dtype:
                device_type=q.device.type
                operand_dtype=(torch.get_autocast_dtype(device_type)
                    if torch.is_autocast_enabled(device_type) else q.dtype)
                accumulated_dtype=(torch.float64 if operand_dtype==torch.float64
                                   else torch.float32)
                local_accumulated=(k.to(operand_dtype).to(accumulated_dtype),
                                   v.to(operand_dtype).to(accumulated_dtype))
        for source,(query_indices,remote_indices) in correspondence["groups"][window].items():
            remote_k,remote_v=projected[source][1:]
            for start in range(0,len(query_indices),query_chunk_size):
                qidx=query_indices[start:start+query_chunk_size]
                ridx=remote_indices[start:start+query_chunk_size]
                if attention_path=="native_sdpa":
                    value,mask_bytes=native_sdpa_attention(q[:,:,qidx],native_buffer,
                        remote_k[:,:,ridx],remote_v[:,:,ridx],True,attention.scale)
                    output[:,:,qidx]=value
                    if memory is not None:
                        memory["native_mask_bytes"]=max(
                            memory.get("native_mask_bytes",0),mask_bytes)
                else:
                    output[:,:,qidx]=paired_softmax_attention(q[:,:,qidx],k,v,
                        remote_k[:,:,ridx],remote_v[:,:,ridx],
                        query_chunk_size=query_chunk_size,
                        local_accumulated=local_accumulated)
        del native_buffer,local_accumulated
        y=attention.proj_drop(attention.proj(output.transpose(1,2).reshape_as(states[window])))
        x=states[window]+block.ls1(y)
        result[window]=x+block.ls2(block.mlp(block.norm2(x)))
    return result
