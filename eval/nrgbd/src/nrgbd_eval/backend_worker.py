"""GPU worker boundary for five model families.

The worker deliberately lives outside the evaluator process. Each native bridge
must return dense world points in selected-frame order; GT is absent from the
request by construction.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
import numpy as np

BRIDGES = {
    "vggt": "nrgbd_eval.native.vggt",
    "vggt_long": "nrgbd_eval.native.vggt_long",
    "streamvggt": "nrgbd_eval.native.streamvggt",
    "vggt_slam": "nrgbd_eval.native.vggt_slam",
    "vggt_omega": "nrgbd_eval.native.vggt_omega",
}


def normalize_dense(points, masks, output_hw=(392, 518)):
    from scipy.ndimage import zoom

    points = np.asarray(points, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    target_h, target_w = map(int, output_hw)
    if points.shape[1:3] == (target_h, target_w):
        return points, masks
    factors = (1, target_h / points.shape[1], target_w / points.shape[2], 1)
    resized = zoom(points, factors, order=1)
    resized_masks = zoom(masks, factors[:3], order=0).astype(bool)
    resized_masks &= np.isfinite(resized).all(-1)
    return resized, resized_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    a = p.parse_args()
    req = json.loads(Path(a.request).read_text())
    name = req["model"]
    if name not in BRIDGES:
        raise ValueError(name)
    sys.path.insert(0, req["source_root"])
    bridge = importlib.import_module(BRIDGES[name])
    out = bridge.infer(req)
    aggregate = out.get("aggregate_points")
    if aggregate is None:
        points = np.asarray(out["world_points"], np.float32)
        masks = np.asarray(out.get("valid_masks", np.isfinite(points).all(-1)), bool)
        if (
            points.ndim != 4
            or points.shape[-1] != 3
            or len(points) != len(req["frame_ids"])
        ):
            raise ValueError(f"invalid world_points shape {points.shape}")
        points, masks = normalize_dense(points, masks, (392, 518))
        payload = {"world_points": points, "valid_masks": masks}
    else:
        aggregate = np.asarray(aggregate, np.float32).reshape(-1, 3)
        payload = {
            "world_points": np.empty((0, 0, 0, 3), np.float32),
            "valid_masks": np.empty((0, 0, 0), bool),
            "aggregate_points": aggregate,
        }
    np.savez_compressed(
        req["output"],
        **payload,
        inference_seconds=np.float64(out["inference_seconds"]),
        peak_allocated_bytes=np.int64(out.get("peak_allocated_bytes", 0)),
        peak_reserved_bytes=np.int64(out.get("peak_reserved_bytes", 0)),
    )


if __name__ == "__main__":
    main()
