# 001 - Atomic Contention

## Intention

Investigate how the distribution of points across voxels affects the performance of GPU voxel counting.

The experiment compares two input distributions with the same number of points and the same maximum voxel occupancy.

## Hypothesis

Even with the same maximum voxel occupancy, a more distributed workload may reduce atomic contention and improve execution time.

## Setup

- Points: 1,000,000
- Voxels: 100,000
- Operation: PyTorch CUDA `index_add_`
- Warmup: 20 runs
- Timed runs: 200

## Input Distribution

- Frame C: 500,000 points in voxel 0 and 500,000 in voxel 1.
- Frame D: 500,000 points in voxel 0; remaining points randomly distributed across voxels 1–99,999.

## Results

| Frame | Time (ms) |
|---|---:|
| C | 1.194270 |
| D | 0.604343 |
