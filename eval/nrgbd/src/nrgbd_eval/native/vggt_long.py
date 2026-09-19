import time
from pathlib import Path
import numpy as np


def infer(req):
    import torch
    from safetensors.torch import load_file
    from loop_utils.config_utils import load_config
    from vggt_long import VGGT_Long
    from base_models.vggt.models.vggt import VGGT

    root = Path(req["source_root"]).resolve()
    project = root.parent if root.name == "base_models" else root
    config = load_config(str(project / "configs" / "base_config.yaml"))
    config["Weights"]["model"] = "VGGT"
    config["Weights"]["VGGT"] = req["checkpoint"]
    config["Model"]["chunk_size"] = int(req["settings"].get("chunk_size", 60))
    config["Model"]["overlap"] = int(req["settings"].get("overlap", 30))
    config["Model"]["loop_enable"] = bool(req["settings"].get("loop_enable", False))
    config["Model"]["delete_temp_files"] = False
    config["Model"]["align_method"] = req["settings"].get("align_method", "numpy")
    work = Path(req["output"]).parent / "vggt_long_runtime"
    runner = VGGT_Long(str(work / "unused"), str(work), config)
    model = VGGT()
    checkpoint = Path(req["checkpoint"])
    if checkpoint.suffix == ".safetensors":
        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    model = model.to(req["device"]).eval()
    elapsed = [0.0]
    original_forward = model.forward

    def timed_forward(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed[0] += time.perf_counter() - started
        return result

    model.forward = timed_forward
    runner.model.model = model
    runner.img_list = list(req["rgb_paths"])
    torch.cuda.reset_peak_memory_stats()
    runner.process_long_sequence()
    point_parts = []
    mask_parts = []
    for index in range(len(runner.chunk_indices)):
        data = np.load(
            work / "_tmp_results_aligned" / f"chunk_{index}.npy",
            allow_pickle=True,
        ).item()
        points = np.asarray(data["world_points"], dtype=np.float32)
        mask = data.get("mask")
        mask = (
            np.asarray(mask).squeeze().astype(bool)
            if mask is not None
            else np.isfinite(points).all(-1)
        )
        keep = slice(None) if index == 0 else slice(config["Model"]["overlap"], None)
        point_parts.append(points[keep])
        mask_parts.append(mask[keep])
    points = np.concatenate(point_parts)
    masks = np.concatenate(mask_parts)
    if len(points) != len(req["frame_ids"]):
        raise RuntimeError(
            f"VGGT-Long exported {len(points)} frames for {len(req['frame_ids'])} inputs"
        )
    return {
        "world_points": points,
        "valid_masks": masks,
        "inference_seconds": elapsed[0],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
