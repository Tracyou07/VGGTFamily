import time
import numpy as np


def infer(req):
    import sys
    import torch

    sys.path[:0] = [str(__import__("pathlib").Path(req["source_root"]) / "src")]
    from streamvggt.models.streamvggt import StreamVGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    model = StreamVGGT()
    state = torch.load(req["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(req["device"]).eval()
    images = load_and_preprocess_images(req["rgb_paths"]).to(req["device"])
    frames = [{"img": im.unsqueeze(0)} for im in images]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        out = model.inference(frames)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t
    points = np.stack(
        [r["pts3d_in_other_view"].float().cpu().numpy()[0] for r in out.ress]
    )
    return {
        "world_points": points,
        "valid_masks": np.isfinite(points).all(-1),
        "inference_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
