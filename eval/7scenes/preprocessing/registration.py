"""Register raw Kinect depth using the SimpleRecon 7-Scenes convention.

Reference: nianticlabs/simplerecon/data_scripts/7scenes_preprocessing.py.
The published transform is a benchmark calibration approximation, not a new
calibration of the individual recording devices. Depth input/output is uint16 mm.
"""
import numpy as np


DEPTH_TO_RGB = np.array([
    [0.99996518012567637, 0.0026765126468950343, -0.0079041012313000904, -0.025558943178152542],
    [-0.00274093112813167, 0.99996302803027592, -0.0081504520778013286, 0.00010109636268061706],
    [0.0078819942130445332, 0.0081718328771890631, 0.99993554554014031, 0.0020318321729487039],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def register_depth(depth_mm, *, depth_focal=585.0, rgb_focal=525.0,
                   principal_point=(320.0, 240.0), transform=None):
    """Project depth pixels into RGB pixels, resolving collisions by nearest Z.

    Preserve the reference's +0.5 source pixel centers, nearest-even rounding
    and truncation back to integer millimeters. Unlike its original script,
    explicitly exclude the dataset's 65535 invalid sentinel before projection.
    """
    depth_mm = np.asarray(depth_mm)
    if depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
        raise ValueError("raw depth must be a 2-D uint16 array in millimeters")
    if depth_focal <= 0 or rgb_focal <= 0:
        raise ValueError("focal lengths must be positive")
    transform = DEPTH_TO_RGB if transform is None else np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("depth-to-RGB transform must be a finite 4x4 matrix")
    height, width = depth_mm.shape
    cx, cy = principal_point
    rows, cols = np.nonzero((depth_mm > 0) & (depth_mm != 65535))
    output = np.zeros_like(depth_mm)
    if not len(rows):
        return output
    depths = depth_mm[rows, cols].astype(np.float32) / np.float32(1000.0)
    points = np.ones((4, len(rows)), dtype=np.float64)
    points[0] = (cols + 0.5 - cx) / depth_focal * depths
    points[1] = (rows + 0.5 - cy) / depth_focal * depths
    points[2] = depths
    rgb_points = transform @ points
    valid = np.isfinite(rgb_points[:3]).all(axis=0) & (rgb_points[2] > 0)
    rgb_points = rgb_points[:, valid]
    u = np.rint(rgb_focal * rgb_points[0] / rgb_points[2] + cx).astype(np.int64)
    v = np.rint(rgb_focal * rgb_points[1] / rgb_points[2] + cy).astype(np.int64)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, v[valid] * width + u[valid], rgb_points[2, valid])
    # The reference stores nearest Z in a float32 image before converting to mm.
    # Retaining float64 here changes integer depths at millimeter boundaries.
    raster_mm = zbuffer.astype(np.float32) * np.float32(1000.0)
    valid = np.isfinite(raster_mm) & (raster_mm < 65535)
    output.reshape(-1)[valid] = raster_mm[valid].astype(np.uint16)
    return output
