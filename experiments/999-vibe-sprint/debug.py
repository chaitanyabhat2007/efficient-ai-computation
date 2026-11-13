import numpy as np, torch
from unproject import unproject
from scenes import SCENES
from cpu_ref import voxelize_ref
import variants

fx = fy = 525.0; cx, cy = 319.5, 239.5
pts, _ = unproject(SCENES["flat_wall"](), fx, fy, cx, cy)
keys, counts, cent = voxelize_ref(pts)
c, m = variants.atomics(torch.from_numpy(pts).cuda())
c = c.cpu().numpy(); m = m.cpu().numpy()
print("points:", len(pts))
print("cpu occupied:", len(counts), "sum:", counts.sum(), "dtype:", counts.dtype, counts.shape)
print("gpu occupied:", len(c), "sum:", c.sum(), "dtype:", c.dtype, c.shape)
print("cpu first 8:", counts[:8])
print("gpu first 8:", c[:8])
if len(c) == len(counts):
    print("mismatching entries:", (c != counts).sum())
    print("centroid max abs diff:", np.abs(m - cent).max())
