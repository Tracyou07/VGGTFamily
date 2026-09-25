"""Per-window states with a barrier before each one-to-one global exchange."""
import torch

from vggt.v6.attention import layout
from vggt.v6.scheduler import initialize
from .attention import MODES, build_correspondences, global_step


def aggregate_windows(aggregator, images, frame_ids, windows, mode,
                      query_chunk_size=16, reverse=False,
                      reuse_image_encoding=False,
                      cache_local_kv_dtype=False,
                      correspondence_attention_path="explicit"):
    if mode not in MODES:
        raise ValueError("unknown v8 mode")
    if len(frame_ids) != len(images):
        raise ValueError("frame IDs must match original input order")
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
    grid = (images.shape[-2] // aggregator.patch_size,
            images.shape[-1] // aggregator.patch_size)
    camera_count, register_count = layout(aggregator)
    correspondence = build_correspondences(
        frame_ids, windows, [frame_ids[lo:hi] for lo, hi in windows],
        [grid] * len(windows), token_count, camera_count, register_count,
        states[0].device, mode)
    channels = states[0].shape[-1]
    features = [[None] * aggregator.depth for _ in windows]
    order = list(range(len(windows)))
    if reverse:
        order.reverse()
    memory = dict(
        window_state_bytes=sum(x.numel() * x.element_size() for x in states)
            + sum(p.numel() * p.element_size() for p in positions if p is not None),
        projected_qkv_bytes=0, head_cache_bytes=0,
        interpretation="tensor-size estimates; not CUDA peak allocated/reserved",
    )
    for layer in range(aggregator.depth):
        frame_cache = {}
        # All frame blocks finish before anyone reads a global-layer bank.
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
                             correspondence, mode, query_chunk_size=query_chunk_size,
                             order=order, memory=memory,
                             cache_local_kv_dtype=cache_local_kv_dtype,
                             attention_path=correspondence_attention_path)
        if layer in aggregator.cached_layer_indices:
            for i, (lo, hi) in enumerate(windows):
                features[i][layer] = torch.cat(
                    [frame_cache[i], states[i].reshape(1, hi - lo, token_count, channels)], dim=-1
                )
    memory["head_cache_bytes"] = sum(
        x.numel() * x.element_size() for window in features for x in window if x is not None
    )
    summary = dict(pair_count=correspondence["pair_count"],
                   patch_grid=correspondence["patch_grid"],
                   tokens_per_frame=token_count,
                   interpretation=correspondence["interpretation"])
    return features, aggregator.patch_start_idx, memory, summary
