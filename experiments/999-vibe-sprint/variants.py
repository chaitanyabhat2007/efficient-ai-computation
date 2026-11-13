import torch
from cpu_ref import VOXEL

def atomics(pts):
    """pts: (N,3) float64 CUDA tensor. Dense grid + index_add_ (atomic adds)."""
    idx = torch.floor(pts / VOXEL).long()
    mn = idx.min(0).values
    idx = idx - mn
    dims = idx.max(0).values + 1
    lin = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    G = int(dims.prod())
    counts = torch.zeros(G, dtype=torch.int64, device=pts.device)
    counts.index_add_(0, lin, torch.ones_like(lin))
    sums = torch.zeros(G, 3, dtype=pts.dtype, device=pts.device)
    sums.index_add_(0, lin, pts)
    occ = counts > 0
    c = counts[occ]
    return c, sums[occ] / c[:, None]
