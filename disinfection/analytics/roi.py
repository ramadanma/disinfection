# motiondetector/analytics/roi.py
import os
import numpy as np


def load_polygon_npy(path: str, name="polygon"):
    """
    Read npy, unify polygons to shape (1, N, 2) with dtype=int32
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 文件不存在: {path}")
    region = np.load(path)
    if len(region.shape) == 2:
        region = region.reshape(1, -1, 2)
    return region.astype(np.int32)


def load_roi_pts(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"road ROI 文件不存在: {path}")
    pts = np.load(path).astype(np.int32)
    return pts
