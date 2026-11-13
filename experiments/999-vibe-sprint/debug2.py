import numpy as np, torch
from unproject import unproject
from scenes import SCENES
from cpu_ref import VOXEL

fx = fy = 525.0; cx, cy = 319.5, 239.5
pts, _ = unproject(SCENES["flat_wall"](), fx, fy, cx, cy)
cpu = np.floor(pts / VOXEL).astype(np.int64)
t = torch.from_numpy(pts).cuda()
a = torch.floor(t / VOXEL).long().cpu().numpy()                      # scalar divide (what variants.py does)
v = torch.tensor(VOXEL, dtype=torch.float64, device="cuda")
b = torch.floor(t / v).long().cpu().numpy()                          # tensor divide
print("points whose voxel differs, scalar divide:", (cpu != a).any(1).sum())
print("points whose voxel differs, tensor divide:", (cpu != b).any(1).sum())
