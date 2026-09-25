"""Opt-in v10 scheduler with unchanged v8 per-window heads and outputs."""
import time
import torch
from torch import nn
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from experiments.ours_v6.windows import make_windows
from .attention import MODES
from .scheduler import aggregate_windows


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def dense_head_cpu(head, cache, images, patch_start, chunk):
    """DPT-only frame batches; concatenate exclusively on CPU.

    DPT uses channel LayerNorm, image-local convolutions/interpolation and
    spatial positional encoding. No operation reduces over frames. Keep the
    shared feature cache alive for both heads; never modify its entries here.
    """
    from vggt.heads.dpt_head import DPTHead
    if not isinstance(head, DPTHead) or head.feature_only:
        raise ValueError("CPU frame batching supports dense DPT prediction heads only")
    predictions, confidences = [], []
    for lo in range(0, images.shape[1], chunk):
        hi = min(lo + chunk, images.shape[1])
        sliced = [None if value is None else value[:, lo:hi] for value in cache]
        pred, conf = head(sliced, images[:, lo:hi], patch_start,
                          frames_chunk_size=None)
        predictions.append(pred.detach().float().cpu())
        confidences.append(conf.detach().float().cpu())
        del pred, conf, sliced
    return torch.cat(predictions, dim=1), torch.cat(confidences, dim=1)


class WindowReconstructor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, images, frame_ids, mode="independent",
                window_size=60, overlap=30, query_chunk_size=16,
                dense_head_frame_chunk=None,
                reuse_image_encoding=False, cache_local_kv_dtype=False,
                correspondence_attention_path="explicit",
                offload_head_features=False,
                stream_projected_qkv=False):
        if dense_head_frame_chunk is not None and dense_head_frame_chunk <= 0:
            raise ValueError("dense head frame chunk must be positive")
        if self.training or self.model.training:
            raise ValueError("v8 requires eval mode")
        if images.ndim != 4 or len(frame_ids) != len(images) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError("one scene with unique frame IDs required")
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("all weights must be frozen")
        if any(getattr(self.model, name, None) is None
               for name in ("camera_head", "depth_head", "point_head")):
            raise ValueError("original camera, depth and point heads required")
        if mode not in MODES:
            raise ValueError("unknown communication mode")
        windows = make_windows(len(images), window_size, overlap)
        device = next(self.model.parameters()).device
        synchronize(device)
        start = time.perf_counter()
        features, patch_start, memory, correspondence = aggregate_windows(
            self.model.aggregator, images, frame_ids, windows, mode,
            query_chunk_size=query_chunk_size,
            reuse_image_encoding=reuse_image_encoding,
            cache_local_kv_dtype=cache_local_kv_dtype,
            correspondence_attention_path=correspondence_attention_path,
            offload_head_features=offload_head_features,
            stream_projected_qkv=stream_projected_qkv,
        )
        synchronize(device)
        backbone_seconds = time.perf_counter() - start
        predictions = []
        head_seconds = 0.0
        transfer_seconds = 0.0
        for i, (lo, hi) in enumerate(windows):
            synchronize(device)
            start = time.perf_counter()
            inputs = images[lo:hi][None].to(device)
            cache = ([None if value is None else value.to(device) for value in features[i]]
                     if offload_head_features else features[i])
            with torch.autocast(device_type=device.type, enabled=False):
                pose_encoding = self.model.camera_head(cache)[-1]
                if dense_head_frame_chunk is None:
                    depth, depth_conf = self.model.depth_head(
                        cache, images=inputs, patch_start_idx=patch_start
                    )
                    points, point_conf = self.model.point_head(
                        cache, images=inputs, patch_start_idx=patch_start
                    )
                else:
                    depth, depth_conf = dense_head_cpu(
                        self.model.depth_head, cache, inputs, patch_start, dense_head_frame_chunk)
                    points, point_conf = dense_head_cpu(
                        self.model.point_head, cache, inputs, patch_start, dense_head_frame_chunk)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_encoding.float(), image_size_hw=inputs.shape[-2:]
                )
                bottom = torch.zeros((*extrinsic.shape[:2], 1, 4),
                                     device=device, dtype=extrinsic.dtype)
                bottom[..., 0, 3] = 1
                c2w = torch.linalg.inv(torch.cat([extrinsic, bottom], dim=-2))
            synchronize(device)
            head_seconds += time.perf_counter() - start
            start = time.perf_counter()
            values = dict(pose_encoding=pose_encoding, c2w=c2w, intrinsics=intrinsic,
                          depth=depth, depth_conf=depth_conf, confidence=depth_conf,
                          world_points=points, world_points_conf=point_conf)
            prediction = {name: value[0].detach().float().cpu().clone()
                          for name, value in values.items()}
            if not all(torch.isfinite(value).all() for value in prediction.values()):
                raise ValueError(f"nonfinite prediction at window {i}")
            if (prediction["intrinsics"][:, [0, 1], [0, 1]] <= 0).any():
                raise ValueError("nonpositive focal length")
            prediction["frame_ids"] = list(frame_ids[lo:hi])
            predictions.append(prediction)
            features[i] = [None] * len(features[i])
            synchronize(device)
            transfer_seconds += time.perf_counter() - start
            del inputs, cache, values, pose_encoding, depth, depth_conf
            del points, point_conf, extrinsic, intrinsic, bottom, c2w
        return dict(predictions=predictions, windows=windows, memory=memory,
                    correspondence=correspondence,
                    timing=dict(backbone_seconds=backbone_seconds,
                                head_seconds=head_seconds,
                                output_transfer_seconds=transfer_seconds),
                    retained_head_features_bytes=memory["head_cache_bytes"],
                    state_device=str(device), cpu_offload=offload_head_features)
