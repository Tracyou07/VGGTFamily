import sys
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch


class NullViewer:
    def __init__(self, *args, **kwargs):
        pass


class NoOpImageRetrieval:
    def get_all_submap_embeddings(self, submap):
        return np.zeros((len(submap.get_all_frames()), 1), dtype=np.float32)

    def find_loop_closures(self, *args, **kwargs):
        return []


def install_noop_salad_import():
    import types

    package = types.ModuleType("salad")
    module = types.ModuleType("salad.eval")
    module.load_model = lambda *args, **kwargs: None
    package.eval = module
    sys.modules["salad"] = package
    sys.modules["salad.eval"] = module


@contextmanager
def offline_dinov2_hub():
    original = torch.hub.load
    local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not local.is_dir():
        raise FileNotFoundError(f"Missing local DINOv2 hub checkout: {local}")

    def load(repo, model, *args, **kwargs):
        if repo == "facebookresearch/dinov2":
            kwargs["source"] = "local"
            repo = str(local)
        return original(repo, model, *args, **kwargs)

    torch.hub.load = load
    try:
        yield
    finally:
        torch.hub.load = original


def load_model(checkpoint, device):
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT

    model = VGGT()
    state = (
        load_file(checkpoint, device="cpu")
        if checkpoint.endswith(".safetensors")
        else torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    class TimedModel(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.total_forward_seconds = 0.0
            self.forward_calls = 0

        def forward(self, images, *args, **kwargs):
            if images.is_cuda:
                torch.cuda.synchronize(images.device)
            started = time.perf_counter()
            with torch.autocast(
                device_type="cuda" if images.is_cuda else "cpu",
                dtype=torch.bfloat16 if images.is_cuda else torch.float32,
                enabled=images.is_cuda,
            ):
                output = self.wrapped(images, *args, **kwargs)
            if images.is_cuda:
                torch.cuda.synchronize(images.device)
            self.total_forward_seconds += time.perf_counter() - started
            self.forward_calls += 1
            return output

    return TimedModel(model).eval()


def iter_submap_windows(paths, size):
    if size <= 0:
        raise ValueError("submap_size must be positive")
    paths = list(paths)
    for start in range(0, len(paths), size):
        window = paths[start : start + size + 1]
        if start > 0 and len(window) == 1:
            break
        if window:
            yield window
