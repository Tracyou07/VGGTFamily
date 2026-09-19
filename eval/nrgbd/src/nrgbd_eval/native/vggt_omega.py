import sys
import time
import numpy as np


def infer(req):
    import torch

    sys.path.insert(0, req["source_root"])
    from run import (
        encoding_to_camera,
        load_and_preprocess_images,
        load_model,
        unproject_depth,
    )

    model = load_model(req["checkpoint"], req["device"])
    images = load_and_preprocess_images(
        list(req["rgb_paths"]),
        mode="balanced",
        image_resolution=512,
        patch_size=16,
    ).to(req["device"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = model(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    depth = predictions["depth"].detach().float().cpu().numpy()
    ext = extrinsic.detach().float().cpu().numpy()
    intr = intrinsic.detach().float().cpu().numpy()
    if depth.shape[0] == 1:
        depth = depth[0]
    if ext.shape[0] == 1:
        ext = ext[0]
    if intr.shape[0] == 1:
        intr = intr[0]
    points = unproject_depth(depth, ext, intr).astype(np.float32)
    conf = predictions["depth_conf"].detach().float().cpu().numpy()
    if conf.shape[0] == 1:
        conf = conf[0]
    return {
        "world_points": points,
        "valid_masks": np.isfinite(points).all(-1) & np.isfinite(conf),
        "inference_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
