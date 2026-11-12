import torch

N = 1_000_000
num_voxels = 100_000
dev = "cuda"

# Frame C: two heavily contended voxels
idx_c = torch.zeros(N, dtype=torch.long, device=dev)
idx_c[N // 2:] = 1

# Frame D: one hot voxel, remaining points spread randomly
idx_d = torch.zeros(N, dtype=torch.long, device=dev)
idx_d[N // 2:] = torch.randint(
    1, num_voxels, (N // 2,), device=dev
)

ones = torch.ones(N, dtype=torch.float32, device=dev)


def count(idx):
    counts = torch.zeros(
        num_voxels, dtype=torch.float32, device=dev
    )
    counts.index_add_(0, idx, ones)
    return counts


def time_it(idx, runs=200):
    for _ in range(20):
        count(idx)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(runs):
        count(idx)

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / runs


print("Frame C ms:", time_it(idx_c))
print("Frame D ms:", time_it(idx_d))
