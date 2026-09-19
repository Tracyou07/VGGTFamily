import time
from pathlib import Path
import numpy as np


def load_images(paths):
    from vggt.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images(paths)


def load_model(req):
    import torch
    from vggt.models.vggt import VGGT

    model = VGGT()
    ck = Path(req["checkpoint"])
    if ck.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ck), device="cpu")
    else:
        state = torch.load(ck, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model.to(req["device"]).eval()


def infer_full(req):
    import torch

    model = load_model(req)
    images = load_images(req["rgb_paths"]).to(req["device"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        out = model(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t
    points = out["world_points"].float().cpu().numpy()[0]
    return {
        "world_points": points,
        "valid_masks": np.isfinite(points).all(-1),
        "inference_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
