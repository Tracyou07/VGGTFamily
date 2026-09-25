"""Shared window definitions; no model or dataset dependencies."""
from collections import defaultdict


def make_windows(count, window_size=30, overlap=10):
    if count < 1 or window_size < 1 or not 0 <= overlap < window_size:
        raise ValueError("require N>0 and 0<=overlap<window_size")
    return tuple((s, min(s + window_size, count))
                 for s in range(0, count, window_size - overlap))


def visibility_groups(count, windows):
    """Partition queries by identical, deduplicated visible frame sets."""
    visible = [set() for _ in range(count)]
    for start, end in windows:
        if not 0 <= start < end <= count:
            raise ValueError("invalid window interval")
        for frame in range(start, end):
            visible[frame].update(range(start, end))
    groups = defaultdict(list)
    for frame, keys in enumerate(visible):
        if not keys:
            raise ValueError("windows must cover every frame")
        groups[tuple(sorted(keys))].append(frame)
    return tuple((tuple(queries), keys) for keys, queries in groups.items())


def slice_features(features, all_ids, requested_ids):
    import torch
    if len(set(all_ids)) != len(all_ids) or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("duplicate frame ID")
    lookup = {frame: i for i, frame in enumerate(all_ids)}
    try:
        indices = [lookup[frame] for frame in requested_ids]
    except KeyError as error:
        raise ValueError("unknown frame ID") from error
    sliced = []
    for value in features:
        if value is None:
            sliced.append(None)
        else:
            if value.shape[1] != len(all_ids):
                raise ValueError("feature/frame count mismatch")
            index = torch.tensor(indices, device=value.device)
            sliced.append(value.index_select(1, index))
    return sliced
