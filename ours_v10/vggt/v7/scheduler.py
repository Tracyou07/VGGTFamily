"""One scene, independent window states, layer-synchronous cross-window exchange."""
import torch

from vggt.v6.scheduler import initialize
from .attention import MODES, global_step, layout
from .sampling import selection_manifest


def aggregate_windows(aggregator, images, windows, mode,
                      patch_exchange_ratio=0.10, query_chunk_size=64, reverse=False,
                      exchange_sdpa_backend="auto", local_query_chunk_size=None,
                      cross_query_chunk_size=None, reuse_image_encoding=False):
    if mode not in MODES:
        raise ValueError("unknown communication mode")
    if reuse_image_encoding:
        # The eval image encoder acts independently on each frame: its ViT
        # attention is spatial within a frame; normalization and position
        # interpolation do not use other frames or window identity.
        device = aggregator.camera_token.device
        all_images = images[None].to(device)
        frames, _, height, width = images.shape
        normalized = (all_images - aggregator._resnet_mean) / aggregator._resnet_std
        patches = aggregator.patch_embed(normalized.reshape(frames, 3, height, width))
        if isinstance(patches, dict):
            patches = patches["x_norm_patchtokens"]
        states, positions, token_count = initialize(
            aggregator, images, windows, precomputed_patch_tokens=patches)
        del patches, normalized, all_images
    else:
        states, positions, token_count = initialize(aggregator, images, windows)
    grid_height = images.shape[-2] // aggregator.patch_size
    grid_width = images.shape[-1] // aggregator.patch_size
    selection = selection_manifest(grid_height, grid_width, patch_exchange_ratio)
    if token_count - aggregator.patch_start_idx != grid_height * grid_width:
        raise ValueError("patch grid differs from actual token count")
    patch_positions = tuple(selection["indices"])
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
        interpretation="live tensor-size estimates for states/bank/QKV/cache; CUDA peak allocated/reserved measured separately",
    )
    for layer in range(aggregator.depth):
        frame_cache = {}
        # Finish all frame blocks before creating the old-state global bank.
        for i in order:
            lo, hi = windows[i]
            frames = hi - lo
            old = states[i].reshape(frames, token_count, channels)
            pos = None if positions[i] is None else positions[i].reshape(frames, token_count, 2)
            updated = aggregator.frame_blocks[layer](old, pos=pos).reshape(1, frames * token_count, channels)
            states[i] = updated
            if layer in aggregator.cached_layer_indices:
                frame_cache[i] = updated.reshape(1, frames, token_count, channels)
        states = global_step(aggregator.global_blocks[layer], states, positions,
                             token_count, mode, camera_count, register_count,
                             patch_positions, query_chunk_size, order=order, memory=memory,
                             exchange_sdpa_backend=exchange_sdpa_backend,
                             local_query_chunk_size=local_query_chunk_size,
                             cross_query_chunk_size=cross_query_chunk_size)
        if layer in aggregator.cached_layer_indices:
            for i, (lo, hi) in enumerate(windows):
                features[i][layer] = torch.cat(
                    [frame_cache[i], states[i].reshape(1, hi - lo, token_count, channels)], dim=-1
                )
    memory["head_cache_bytes"] = sum(
        x.numel() * x.element_size() for window in features for x in window if x is not None
    )
    return features, aggregator.patch_start_idx, memory, selection
