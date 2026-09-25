"""Deterministic, exact-count, spatially spread patch selection (farthest_grid_v1)."""
import math
from functools import lru_cache

import numpy as np

ALGORITHM = "farthest_grid_v1"


@lru_cache(maxsize=128)
def select_patch_positions(height, width, ratio):
    height, width = int(height), int(width)
    ratio = float(ratio)
    if height < 1 or width < 1 or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("positive patch grid and finite ratio in [0, 1] required")
    count = min(height * width, math.ceil(ratio * height * width))
    if count == 0:
        return ()
    if count == height * width:
        return tuple(range(count))
    rows, cols = np.indices((height, width))
    coords = np.stack(((rows.ravel() + .5) / height,
                       (cols.ravel() + .5) / width), axis=1)
    center = np.array([.5, .5])
    selected = []
    nearest = np.full(height * width, np.inf)
    # The center starts the deterministic farthest-point sweep; argmax breaks ties
    # by lowest flat row-major index. No rounding or deduplication changes count.
    current = int(np.argmin(((coords - center) ** 2).sum(axis=1)))
    for _ in range(count):
        selected.append(current)
        distance = ((coords - coords[current]) ** 2).sum(axis=1)
        np.minimum(nearest, distance, out=nearest)
        nearest[selected] = -1
        current = int(np.argmax(nearest))
    return tuple(sorted(selected))


def selection_manifest(height, width, ratio):
    indices = select_patch_positions(height, width, ratio)
    return dict(algorithm=ALGORITHM, grid_height=int(height), grid_width=int(width),
                requested_ratio=float(ratio), selected_count=len(indices),
                total_patches=int(height) * int(width),
                actual_ratio=len(indices) / (int(height) * int(width)),
                indices=list(indices), index_order="row-major, zero-based")
