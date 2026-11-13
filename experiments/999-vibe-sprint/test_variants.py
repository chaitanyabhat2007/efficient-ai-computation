import numpy as np, torch
from unproject import unproject
from scenes import SCENES
from cpu_ref import voxelize_ref
import variants

fx = fy = 525.0; cx, cy = 319.5, 239.5
for name, fn in SCENES.items():
    pts, _ = unproject(fn(), fx, fy, cx, cy)
    keys, counts, cent = voxelize_ref(pts)
    c, m = variants.atomics(torch.from_numpy(pts).cuda())
    c, m = c.cpu().numpy(), m.cpu().numpy()
    ok_counts = np.array_equal(c, counts)
    ok_cent = np.allclose(m, cent, rtol=1e-6, atol=1e-9) if len(c) == len(counts) else False
    print(f"{name:16s} counts_exact={ok_counts} centroids_ok={ok_cent}")
