
import time
import statistics
import csv


# Different input sizes for our experiment
INPUT_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]

# Number of times to repeat measurements
WARMUP_RUNS = 5
MEASUREMENT_RUNS = 30


# Method 1: Calculate the sum using a for loop
def sum_using_loop(numbers):
    total = 0

    for number in numbers:
        total += number

    return total


# Method 2: Calculate the sum using Python's built-in sum()
def sum_using_builtin(numbers):
    return sum(numbers)


# Measure how long a function takes to execute
def measure_execution_time(function, numbers):
    start_time = time.perf_counter_ns()

    function(numbers)

    end_time = time.perf_counter_ns()

    execution_time = end_time - start_time

    return execution_time


# Run the experiment
def main():

    print("EXECUTION TIME BENCHMARK")
    print("------------------------")

    # Store all individual measurements
    results = []

    for size in INPUT_SIZES:

        # Create a list of numbers
        numbers = list(range(size))

        # Verify correctness before benchmarking
        loop_result = sum_using_loop(numbers)
        builtin_result = sum_using_builtin(numbers)

        expected_result = size * (size - 1) // 2

        assert loop_result == builtin_result, "The two methods returned different results!"
        assert loop_result == expected_result, "Incorrect summation result!"

        # Warm-up runs (not included in measurements)
        for _ in range(WARMUP_RUNS):
            sum_using_loop(numbers)
            sum_using_builtin(numbers)

        # Measure the loop method
        loop_times = []

        for run in range(MEASUREMENT_RUNS):
            elapsed = measure_execution_time(sum_using_loop, numbers)
            loop_times.append(elapsed)

            results.append({
                "input_size": size,
                "method": "loop",
                "run": run + 1,
                "execution_time_ns": elapsed
            })

        # Measure the built-in method
        builtin_times = []

        for run in range(MEASUREMENT_RUNS):
            elapsed = measure_execution_time(sum_using_builtin, numbers)
            builtin_times.append(elapsed)

            results.append({
                "input_size": size,
                "method": "builtin",
                "run": run + 1,
                "execution_time_ns": elapsed
            })

        # Calculate statistics
        loop_median = statistics.median(loop_times)
        builtin_median = statistics.median(builtin_times)

        loop_min = min(loop_times)
        builtin_min = min(builtin_times)

        loop_max = max(loop_times)
        builtin_max = max(builtin_times)

        # Display results
        print(f"\nInput size: {size:,}")

        print(
            f"Loop method: "
            f"Median = {loop_median:,} ns, "
            f"Min = {loop_min:,} ns, "
            f"Max = {loop_max:,} ns"
        )

        print(
            f"Built-in sum: "
            f"Median = {builtin_median:,} ns, "
            f"Min = {builtin_min:,} ns, "
            f"Max = {builtin_max:,} ns"
        )

    # Save all measurements to CSV
    output_file = "experiments/001_execution_time/results.csv"

    with open(output_file, "w", newline="") as file:
        fieldnames = [
            "input_size",
            "method",
            "run",
            "execution_time_ns"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
