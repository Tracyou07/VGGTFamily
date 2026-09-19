from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ModelSceneInput:
    scene_id: str
    frame_ids: tuple[str, ...]
    rgb_paths: tuple[Path, ...]

    def __post_init__(self):
        if len(self.frame_ids) != len(self.rgb_paths) or not self.frame_ids:
            raise ValueError("non-empty aligned frame ids and RGB paths required")
        if any(not p.is_absolute() for p in self.rgb_paths):
            raise ValueError("absolute RGB paths required")


@dataclass
class SceneInput:
    model: ModelSceneInput
    depth_paths: tuple[Path, ...]
    poses_c2w: np.ndarray
    intrinsics: np.ndarray
    resize_shapes: tuple[tuple[int, int, int, int], ...]

    def depth_m(self, index):
        from PIL import Image

        im = Image.open(self.depth_paths[index])
        rw, rh, left, top = self.resize_shapes[index]
        im = im.resize((rw, rh), resample=Image.Resampling.NEAREST)
        im = im.crop((left, top, left + 518, top + 392))
        d = np.asarray(im, dtype=np.float32) / 1000.0
        d[(d < 1e-3) | (d > 10) | ~np.isfinite(d)] = 0
        return d


@dataclass
class ScenePrediction:
    scene_id: str
    frame_ids: tuple[str, ...]
    world_points: np.ndarray
    valid_masks: np.ndarray
    inference_seconds: float
    adapter_seconds: float
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    metadata: dict = field(default_factory=dict)
    aggregate_points: np.ndarray | None = None
