import numpy as np


def backproject(depth, K, c2w):
    y, x = np.indices(depth.shape)
    z = depth
    cam = np.stack(
        ((x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)),
        axis=-1,
    )
    return (cam @ c2w.T)[..., :3]


def scale_shift_align(pred, gt, mask):
    p = np.asarray(pred)[mask]
    g = np.asarray(gt)[mask]
    pc = p - p.mean(0)
    gc = g - g.mean(0)
    den = float((pc * pc).sum())
    if den <= 1e-12:
        raise ValueError("degenerate predicted geometry")
    scale = float((pc * gc).sum() / den)
    shift = g.mean(0) - scale * p.mean(0)
    return scale * np.asarray(pred) + shift


def transform_points(points, transform):
    points = np.asarray(points)
    return (
        np.einsum("ij,...j->...i", np.asarray(transform)[:3, :3], points)
        + np.asarray(transform)[:3, 3]
    )
