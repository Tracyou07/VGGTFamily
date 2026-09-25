"""Per-window execution with a layer barrier before every cross-window update."""
import torch
from vggt.models.aggregator import slice_expand_and_flatten
from .attention import MODES, global_step, layout, special_indices, _checksum


def initialize(aggregator, images, windows, precomputed_patch_tokens=None):
    if aggregator.training or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("eval aggregator and one [N,3,H,W] scene required")
    if aggregator.aa_order != ["frame", "global"] or aggregator.aa_block_size != 1:
        raise ValueError("unsupported backbone layer order")
    if not windows:
        raise ValueError("empty window list")
    covered = set()
    for lo, hi in windows:
        if not 0 <= lo < hi <= len(images):
            raise ValueError("window outside scene")
        covered.update(range(lo, hi))
    if covered != set(range(len(images))):
        raise ValueError("windows must cover all real frames")
    device = aggregator.camera_token.device
    if precomputed_patch_tokens is not None:
        if (precomputed_patch_tokens.ndim != 3 or
                precomputed_patch_tokens.shape[0] != len(images) or
                precomputed_patch_tokens.device != device):
            raise ValueError("cached image patches must match all frames and model device")
    states, positions = [], []
    camera_count, register_count = layout(aggregator)
    for lo, hi in windows:
        # No window batch dimension greater than one is ever constructed.
        frames = hi - lo
        height, width = images.shape[-2:]
        if precomputed_patch_tokens is None:
            window_input = images[lo:hi][None].to(device)
            normalized = (window_input - aggregator._resnet_mean) / aggregator._resnet_std
            patch = aggregator.patch_embed(normalized.reshape(frames, 3, height, width))
            if isinstance(patch, dict):
                patch = patch["x_norm_patchtokens"]
        else:
            patch = precomputed_patch_tokens[lo:hi]
        camera = slice_expand_and_flatten(aggregator.camera_token, 1, frames)
        register = slice_expand_and_flatten(aggregator.register_token, 1, frames)
        tokens = torch.cat([camera, register, patch], dim=1)
        token_count, channels = tokens.shape[-2:]
        pos = None
        if aggregator.rope is not None:
            spatial = aggregator.position_getter(
                frames, height // aggregator.patch_size, width // aggregator.patch_size, device=device
            ) + 1
            special = torch.zeros(frames, aggregator.patch_start_idx, 2,
                                  device=device, dtype=spatial.dtype)
            pos = torch.cat([special, spatial], dim=1).reshape(1, frames * token_count, 2)
        states.append(tokens.reshape(1, frames * token_count, channels).clone())
        positions.append(pos)
    if camera_count + register_count != aggregator.patch_start_idx:
        raise ValueError("model special-token layout changed")
    return states, positions, token_count


def aggregate_windows(aggregator, images, windows, mode,
                      query_chunk_size=64, reverse=False, diagnostics=None, frame_ids=None):
    if mode not in MODES:
        raise ValueError("unknown communication mode")
    states, positions, token_count = initialize(aggregator, images, windows)
    if frame_ids is not None and len(frame_ids) != len(images):
        raise ValueError("diagnostic frame IDs must match image count")
    camera_count, register_count = layout(aggregator)
    channels = states[0].shape[-1]
    features = [[None] * aggregator.depth for _ in windows]
    order = list(range(len(windows)))
    if reverse:
        order.reverse()
    memory = dict(
        window_state_bytes=sum(x.numel() * x.element_size() for x in states)
            + sum(p.numel() * p.element_size() for p in positions if p is not None),
        communication_bank_bytes=0, head_cache_bytes=0,
        temporary_qkv_bytes=max(x.numel() * x.element_size() * 3 for x in states),
        concatenated_kv_bytes=0,
        interpretation="live tensor-size estimates for states/bank/QKV/cache; CUDA peak allocated/reserved are measured separately",
    )
    def register_state(state):
        indices = special_indices(state.shape[1], token_count,
                                  camera_count + register_count, state.device)
        indices = indices.reshape(-1, camera_count + register_count)[:, camera_count:].reshape(-1)
        value = state[:, indices]
        return float(value.float().norm(dim=-1).mean()), _checksum(value)

    previous_rows = None
    for layer in range(aggregator.depth):
        if diagnostics is not None and previous_rows is not None:
            for row in previous_rows:
                norm, checksum = register_state(states[row["window_index"]])
                row["next_layer_register_input_norm"] = norm
                row["next_layer_register_input_checksum"] = checksum
                row["register_writeback_preserved"] = checksum == row["register_output_checksum"]
        frame_cache = {}
        # Every frame block completes before any window enters this layer's global block.
        for i in order:
            lo, hi = windows[i]
            frames = hi - lo
            old = states[i].reshape(frames, token_count, channels)
            pos = None if positions[i] is None else positions[i].reshape(frames, token_count, 2)
            updated = aggregator.frame_blocks[layer](old, pos=pos).reshape(1, frames * token_count, channels)
            states[i] = updated
            if layer in aggregator.cached_layer_indices:
                frame_cache[i] = updated.reshape(1, frames, token_count, channels)
        input_register_norms = None
        if diagnostics is not None:
            input_register_norms = [register_state(state)[0] for state in states]
        states = global_step(aggregator.global_blocks[layer], states, positions,
                             token_count, mode, camera_count, register_count,
                             query_chunk_size, order=order, memory=memory,
                             diagnostics=diagnostics, layer=layer)
        if diagnostics is not None:
            previous_rows = diagnostics[-len(windows):]
            for row in previous_rows:
                i = row["window_index"]
                lo, hi = windows[i]
                row["window_range"] = [lo, hi]
                row["frame_ids"] = list(frame_ids[lo:hi]) if frame_ids is not None else list(range(lo, hi))
                row["register_input_norm"] = input_register_norms[i]
                row["register_output_norm"], row["register_output_checksum"] = register_state(states[i])
                row["register_remote_delta_norm"] = row["query_types"]["register"]["query_delta_norm"]
        if layer in aggregator.cached_layer_indices:
            for i, (lo, hi) in enumerate(windows):
                features[i][layer] = torch.cat(
                    [frame_cache[i], states[i].reshape(1, hi - lo, token_count, channels)], dim=-1
                )
    memory["head_cache_bytes"] = sum(
        x.numel() * x.element_size() for window in features for x in window if x is not None
    )
    return features, aggregator.patch_start_idx, memory
