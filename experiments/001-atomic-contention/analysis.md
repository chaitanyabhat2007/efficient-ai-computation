# Analysis

Frame D completed in approximately half the time of Frame C.

| Frame | Time (ms) |
|---|---:|
| C | 1.194270 |
| D | 0.604343 |

Both frames have:
- 1,000,000 total points.
- A maximum voxel occupancy of 500,000 points.

The difference is in the distribution of the remaining points.

Frame C concentrates them in one additional voxel, while Frame D distributes them randomly across many voxels.

This suggests that maximum voxel occupancy alone does not explain the performance difference. The overall distribution of updates also matters.
