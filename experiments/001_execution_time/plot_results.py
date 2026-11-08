
import csv
import statistics
from pathlib import Path
import matplotlib.pyplot as plt


# Locate the experiment directory
EXPERIMENT_DIR = Path(__file__).parent
CSV_FILE = EXPERIMENT_DIR / "results.csv"
PLOTS_DIR = EXPERIMENT_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


# Read measurements from the CSV file
measurements = {}

with open(CSV_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        size = int(row["input_size"])
        method = row["method"]
        execution_time = int(row["execution_time_ns"])

        key = (size, method)

        if key not in measurements:
            measurements[key] = []

        measurements[key].append(execution_time)


# Calculate median execution time for each method and input size
input_sizes = sorted(set(size for size, method in measurements))

loop_medians = []
builtin_medians = []

for size in input_sizes:
    loop_median = statistics.median(measurements[(size, "loop")])
    builtin_median = statistics.median(measurements[(size, "builtin")])

    loop_medians.append(loop_median)
    builtin_medians.append(builtin_median)


# Plot 1: Execution time vs input size
plt.figure(figsize=(9, 6))

plt.plot(input_sizes, loop_medians, marker="o", label="Manual loop")
plt.plot(input_sizes, builtin_medians, marker="o", label="Built-in sum()")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Input size (number of elements)")
plt.ylabel("Median execution time (nanoseconds)")
plt.title("Execution Time vs Input Size")

plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "execution_time_comparison.png", dpi=300)
plt.show()


# Plot 2: Speedup vs input size
speedups = [
    loop_time / builtin_time
    for loop_time, builtin_time in zip(loop_medians, builtin_medians)
]

plt.figure(figsize=(9, 6))

plt.plot(input_sizes, speedups, marker="o")

plt.xscale("log")

plt.xlabel("Input size (number of elements)")
plt.ylabel("Speedup (loop time / built-in sum time)")
plt.title("Built-in sum() Speedup Over Manual Loop")

plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "speedup_comparison.png", dpi=300)
plt.show()


print("Plots saved successfully!")
