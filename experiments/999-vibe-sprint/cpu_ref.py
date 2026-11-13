import numpy as np
from unproject import unproject
from scenes import SCENES

VOXEL = 0.05  # 5 cm

def voxelize_ref(pts):
    """float64 reference. Voxel = floor(p / VOXEL), half-open cells [k*s, (k+1)*s)."""
    idx = np.floor(pts / VOXEL).astype(np.int64)
    keys, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    cent = np.stack([np.bincount(inv, weights=pts[:, d]) for d in range(3)], axis=1) / counts[:, None]
    return keys, counts, cent

if __name__ == "__main__":
    fx = fy = 525.0; cx, cy = 319.5, 239.5
    for name, fn in SCENES.items():
        pts, _ = unproject(fn(), fx, fy, cx, cy)
        keys, counts, cent = voxelize_ref(pts)
        print(f"{name:16s} points={len(pts):7d} occupied={len(keys):6d} "
              f"max={counts.max():5d} mean={counts.mean():7.1f}")
