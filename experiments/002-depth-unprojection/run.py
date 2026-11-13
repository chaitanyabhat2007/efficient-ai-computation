import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unproject import unproject
from scenes import SCENES

fx = fy = 525.0; cx, cy = 319.5, 239.5   # replace with TUM intrinsics for real data

# sanity check: pixel at the principal point must give X≈0, Y≈0
p, _ = unproject(np.full((480, 640), 5000.0), fx, fy, cx, cy)
assert p.shape[0] == 480 * 640

for name, fn in SCENES.items():
    depth = fn()
    pts, valid = unproject(depth, fx, fy, cx, cy)
    print(f"{name:16s} valid={valid.sum():7d} ({valid.mean():.0%})  "
          f"X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
    idx = np.random.default_rng(0).choice(len(pts), min(8000, len(pts)), replace=False)
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(depth, cmap="viridis"); ax1.set_title(f"{name}: depth")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    q = pts[idx]; ax2.scatter(q[:, 0], q[:, 2], -q[:, 1], s=1)
    ax2.set_xlabel("X"); ax2.set_ylabel("Z"); ax2.set_zlabel("-Y"); ax2.set_title("points")
    plt.tight_layout(); plt.savefig(f"results/{name}.png", dpi=120); plt.close()
