# Experiment 003: Atomic Voxel Latency

## Question

As more points pile into fewer voxels, how does the latency of atomic-based aggregation change, and why?

Specifically, at what level of concentration does contention on a single voxel begin to dominate the kernel's runtime, and how does latency grow beyond that point?

## Hypothesis

Latency will rise as points concentrate into fewer voxels, because `atomicAdd`s to the same address are serialized at the L2, so the busiest voxel forms a queue of length N that takes roughly N × T to drain.

The rise will not start immediately. At low concentration, the parallel work, roughly (P / K) × T, dominates and the latency will stay flat. Once N exceeds P / K, the queue takes over and latency will climb roughly linearly with N.

I expect a hockey-stick shape: a flat stretch, then a steep, roughly linear rise. It will probably sit slightly above the queue estimate because of fixed overheads and contention costs. I do not expect true exponential growth, since the worst case is bounded by P serialized atomics.

## Theory

### What an atomicAdd does

Each thread handles one point, works out which voxel it belongs to, and calls `atomicAdd` on that voxel's slot in global memory. The instruction does not complete inside the streaming multiprocessor (SM). The thread sends a small request packet containing the address, the operation, and the value out to the L2 cache. The L2 is used because shared memory is private to each SM, while the L2 is the one place every SM can reach.

### Where delay can occur

There are three costs:

1. The trip from the SM to the L2 and back, which is roughly fixed.
2. The queue of requests waiting at the L2 for the same address.
3. The handling time per atomic, which is the time to read, add, and write back.

Only the queue grows with pileup.

### Serialization

Atomics to the same address cannot overlap, so they run one after another. Atomics to different addresses can run in parallel.

### Two competing terms

Let:

- **T** be the handling time per atomic,
- **N** be the number of points in the busiest voxel,
- **P** be the total number of points,
- **K** be the number of atomics the GPU can process at once.

The hottest voxel takes about N × T. The parallel work takes about (P / K) × T. Latency is roughly the larger of the two, plus fixed overhead:

```
latency ≈ max(N × T, (P / K) × T) + overhead
```

### The knee

The crossing happens when N = P / K. Before it, latency is flat. Above it, latency grows roughly linearly with N.

### Caveats

K is an effective number, not a spec. Real hardware adds costs this model ignores, such as warp-level conflicts and several voxels sharing an L2 slice.

## Graph shape candidates

Axes: latency (ms) on the y-axis, average points per voxel on the x-axis.

1. **Flat, then linear.** Latency stays flat while parallel work dominates, then climbs in a straight line once the queue takes over. This is my main prediction.
2. **Smooth, gradual bend.** There is no sharp knee because queueing and parallel work overlap, so the transition is soft.
3. **Staircase.** Latency jumps at certain concentrations, followed by plateaus, caused by hardware boundaries such as warp size or cache limits.
4. **Rise, then saturate.** Latency climbs and then levels off because the hardware combines or merges same-address atomics, for example within a warp, so extreme pileup stops getting worse.
5. **Exponential.** Latency doubles with each fixed step of concentration. I consider this unlikely since the worst case is bounded by P serialized atomics. Note that a straight line on a linear axis becomes an exponential-looking curve on a log axis, so axis choice matters when reading the plot.

**My pick is shape 1, with shape 4 as the main alternative.**

## Variables

### Varied (independent variable)

Points per voxel, i.e. the concentration. I sweep it from the uniform baseline up to all points in a single voxel, in roughly logarithmic steps.

### Fixed (for now)

- Total number of points: 1 million.
- Voxel grid: 64 × 64 × 64, flat 1D layout in global memory.
- Point order in memory: shuffled.
- Distribution shape: uniform at first.
- Operation: `atomicAdd` on `int32` counts.
- Block size: 256 (grid size derived from it).
- Random seed, GPU, driver, CUDA version, compiler, and flags.

### Measured

Kernel latency in milliseconds, timed with CUDA events. After warmup, each level is repeated about 100 times, and I report the median and spread. The grid is cleared before every run, outside the timed region.

### Later experiments (one change at a time)

Total points, point order, data type, block size, distribution shape, and shared-memory privatization.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| Compute capability | 12.0 |
| Streaming multiprocessors | 36 |
| L2 cache size | 32 MiB (33,554,432 bytes) |
| Driver version | 595.99.02 |
| CUDA version (reported by driver) | 13.2 |
| CUDA toolkit (`nvcc`) | 13.4 (V13.4.92) |
| Compiler flags | TODO |
| Python | 3.12.14 |
| Kernel launch library | TODO |
| Operating system | Ubuntu 24.04.5 LTS |
| GPU shared with display? | Yes. Xorg, gnome-shell and other desktop processes use ~409 MiB |

The GPU also drives the desktop, and it idles at a low power state (P8) until it gets work. Both are possible sources of noise, which is why the warmup step and the randomized run order matter.

## Methods

1. **Generate the workload.** For each concentration level, build 1 million points whose voxel indices are drawn so that the average points per voxel hits the target. Then shuffle the order. The seed is fixed.
2. **Allocate and clear.** Create the voxel grid in GPU memory and zero it before every run, outside the timed region.
3. **Warm up.** Run the kernel a few times, untimed, so compilation and clock ramp-up do not pollute the results.
4. **Time.** Record a start event, launch the `atomicAdd` kernel, record a stop event, and read the elapsed milliseconds. Repeat about 100 times.
5. **Record.** Save the median, the spread, and the maximum number of points in any single voxel, because the max is what the theory says matters.
6. **Randomize the run order** of the concentration levels, so any drift in temperature or clock speed spreads out instead of looking like a trend.
7. **Verify correctness.** After each run, check that the voxel counts sum to the total number of points. This catches silent bugs.

## How to run

```bash
python benchmark.py
```

## Results

_To be filled in after the experiment._

## Conclusion

_To be filled in after the experiment._