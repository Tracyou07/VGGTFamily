"""Reproducible local point-head correspondence masks for the unchanged Long fitter."""
from pathlib import Path
import numpy as np
from vggt.v5.alignment import validate_prediction


def save_overlap_mask(previous, current, path):
    a_ids, _, a_conf = validate_prediction(previous)
    b_ids, _, b_conf = validate_prediction(current)
    b_lookup = {frame: index for index, frame in enumerate(b_ids)}
    common = [frame for frame in a_ids if frame in b_lookup]
    if not common:
        raise ValueError("no common frame IDs")
    a_index = [a_ids.index(frame) for frame in common]
    b_index = [b_lookup[frame] for frame in common]
    ac = a_conf[a_index]
    bc = b_conf[b_index]
    if ac.shape != bc.shape:
        raise ValueError("overlap pixel grids differ")
    threshold = float(0.1 * min(np.median(ac), np.median(bc)))
    valid = (ac > threshold) & (bc > threshold)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, common_frame_ids=np.asarray(common),
                        preceding_local_index=np.asarray(a_index),
                        following_local_index=np.asarray(b_index),
                        confidence_threshold=np.asarray(threshold),
                        mask_shape=np.asarray(valid.shape),
                        packed_valid_mask=np.packbits(valid.reshape(-1)),
                        valid_correspondences=np.asarray(valid.sum()))
    return int(valid.sum())
