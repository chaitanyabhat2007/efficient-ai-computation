import numpy as np

def unproject(depth_raw, fx, fy, cx, cy, scale=5000.0):
    """depth_raw: (H,W) raw depth. Returns (N,3) float64 points and valid mask."""
    H, W = depth_raw.shape
    Z = depth_raw.astype(np.float64) / scale
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = Z > 0
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack([X[valid], Y[valid], Z[valid]], axis=1), valid
