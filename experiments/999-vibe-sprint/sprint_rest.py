import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unproject import unproject
from scenes import SCENES
from cpu_ref import voxelize_ref, VOXEL

# ---------- shared voxel index (ONE place, so all variants agree on boundaries) ----------
def voxel_lin(pts):
    vs = torch.tensor(VOXEL, dtype=pts.dtype, device=pts.device)  # tensor divide: fixes the boundary bug
    idx = torch.floor(pts / vs).long()
    idx = idx - idx.min(0).values
    dims = idx.max(0).values + 1
    lin = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    return lin, int(dims.prod())

# ---------- variant 1: atomics (index_add_ on a dense grid) ----------
def atomics(pts):
    lin, G = voxel_lin(pts)
    counts = torch.zeros(G, dtype=torch.int64, device=pts.device)
    counts.index_add_(0, lin, torch.ones_like(lin))
    sums = torch.zeros(G, 3, dtype=pts.dtype, device=pts.device)
    sums.index_add_(0, lin, pts)
    occ = counts > 0
    c = counts[occ]
    return c, sums[occ] / c[:, None]

# ---------- variant 2: sort, then reduce segments (no atomics) ----------
def sort_reduce(pts):
    lin, _ = voxel_lin(pts)
    lin_s, perm = torch.sort(lin)
    ps = pts[perm]
    _, counts = torch.unique_consecutive(lin_s, return_counts=True)
    cs = torch.cat([torch.zeros(1, 3, dtype=pts.dtype, device=pts.device), torch.cumsum(ps, 0)])
    ends = torch.cumsum(counts, 0)
    sums = cs[ends] - cs[ends - counts]
    return counts, sums / counts[:, None]

# ---------- variant 3: torch.bincount (PyTorch's built-in histogram kernel) ----------
def bincount(pts):
    lin, G = voxel_lin(pts)
    counts = torch.bincount(lin, minlength=G)
    sums = torch.stack([torch.bincount(lin, weights=pts[:, d], minlength=G).to(pts.dtype)
                        for d in range(3)], dim=1)
    occ = counts > 0
    c = counts[occ]
    return c, sums[occ] / c[:, None]

VARIANTS = {"atomics": atomics, "sort_reduce": sort_reduce, "bincount": bincount}

# ---------- step 4: correctness vs float64 CPU reference ----------
def check_correctness():
    fx = fy = 525.0; cx, cy = 319.5, 239.5
    all_ok = True
    for name, fn in SCENES.items():
        pts, _ = unproject(fn(), fx, fy, cx, cy)
        _, counts, cent = voxelize_ref(pts)
        t = torch.from_numpy(pts).cuda()
        for vname, v in VARIANTS.items():
            c, m = v(t)
            c, m = c.cpu().numpy(), m.cpu().numpy()
            ok_c = np.array_equal(c, counts)
            ok_m = len(c) == len(counts) and np.allclose(m, cent, rtol=1e-6, atol=1e-9)
            all_ok &= ok_c and ok_m
            print(f"{name:16s} {vname:12s} counts_exact={ok_c} centroids_ok={ok_m}")
    return all_ok

# ---------- step 5: timing harness (CUDA events, warmup, percentiles) ----------
def time_variant(fn, pts, warm=5, iters=50):
    for _ in range(warm):
        fn(pts)
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(pts); e.record(); torch.cuda.synchronize()
        ms.append(s.elapsed_time(e))
    return np.percentile(ms, [50, 95, 99])

# ---------- step 6: synthetic workloads (valid fraction x clustering) ----------
def synth(valid_frac, cluster, seed=0, total=640 * 480):
    rng = np.random.default_rng(seed)
    n = int(total * valid_frac); nh = int(n * cluster)
    uni = rng.uniform([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5], (n - nh, 3))
    hot = rng.normal([0.0, 0.0, 2.0], 0.05, (nh, 3))   # dense blob = heavy contention
    return np.vstack([uni, hot]).astype(np.float32)

if __name__ == "__main__":
    print("== correctness (float64 vs CPU reference) ==")
    assert check_correctness(), "A variant disagrees with the reference. Fix before trusting timings."

    valids = [0.1, 0.3, 0.5, 0.7, 0.9]
    clusters = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
    names = list(VARIANTS)
    win = np.zeros((len(valids), len(clusters)), dtype=int)
    best_ms = np.zeros_like(win, dtype=float)
    rows = ["valid,cluster," + ",".join(f"{n}_p50,{n}_p95,{n}_p99" for n in names)]
    print("\n== timing (float32, ms) ==")
    for i, vf in enumerate(valids):
        for j, cl in enumerate(clusters):
            t = torch.from_numpy(synth(vf, cl)).cuda()
            res = {n: time_variant(VARIANTS[n], t) for n in names}
            p95 = [res[n][1] for n in names]
            win[i, j] = int(np.argmin(p95)); best_ms[i, j] = min(p95)
            rows.append(f"{vf},{cl}," + ",".join(f"{x:.4f}" for n in names for x in res[n]))
            print(f"valid={vf:.1f} cluster={cl:.2f} " +
                  " ".join(f"{n}={res[n][1]:.3f}" for n in names) + f"  -> {names[win[i, j]]}")
    open("results/timings.csv", "w").write("\n".join(rows))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(win, cmap="Set2", vmin=0, vmax=7, aspect="auto")
    for i in range(len(valids)):
        for j in range(len(clusters)):
            ax.text(j, i, f"{names[win[i, j]]}\n{best_ms[i, j]:.2f} ms", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels(clusters)
    ax.set_yticks(range(len(valids))); ax.set_yticklabels(valids)
    ax.set_xlabel("fraction of points in one hot blob (clustering)")
    ax.set_ylabel("valid-pixel fraction")
    ax.set_title("Fastest variant by p95 latency")
    plt.tight_layout(); plt.savefig("results/heatmap.png", dpi=130)
    print("\nSaved results/heatmap.png and results/timings.csv")
