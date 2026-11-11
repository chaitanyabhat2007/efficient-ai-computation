# Experiment 001 — Analysis

## 1. Understanding Python and CPython

I started by understanding what Python actually is and how a Python program gets executed.

Python is a programming language, while CPython is an implementation of that language. In simple terms, Python defines how the language should behave, and CPython is one of the programs that makes that behavior happen.

CPython does not directly execute the Python source code as written. It first compiles the source code into bytecode. This bytecode consists of instructions that CPython's interpreter can understand. Compiled bytecode can also be cached in `.pyc` files.

The interpreter then executes these instructions using its evaluation loop, which is implemented in C.

## 2. How CPython Executes a Program

I understood the execution process as a sequence of stages:

1. CPython reads the Python source code.
2. Its compiler processes the source code and generates bytecode.
3. The Python interpreter starts executing the bytecode instructions.
4. During execution, the interpreter accesses variables and objects managed by the Python runtime.

The interpreter repeatedly processes bytecode instructions. Conceptually, this involves fetching an instruction, determining what operation it represents, and executing that operation.

I also learned that variables do not simply contain their values directly. Python variables refer to objects managed by the runtime. When an instruction needs a variable, the interpreter resolves that variable and accesses the corresponding object.

## 3. Why Built-in `sum()` Is Faster Than a Manual Loop

This was the main concept I wanted to understand in this experiment.

I learned that the difference lies in how the operations are executed.

With a manual `for` loop, the loop body is represented by bytecode instructions. These instructions are repeatedly executed by the Python interpreter for every iteration. This introduces interpreter overhead, along with the overhead of Python-level operations.

The built-in `sum()` function works differently. In CPython, its implementation is written in C. Python still needs to execute the function call, but once the call enters the built-in implementation, the iteration and accumulation happen inside compiled C code.

This avoids repeatedly executing the summation logic through Python-level bytecode instructions.

Therefore, even though both approaches perform the same fundamental summation work, the built-in function has significantly less interpreter overhead.

**My understanding:** The performance difference is not because `sum()` performs less fundamental summation work. It is mainly because it performs that work at a lower level, avoiding much of the overhead associated with repeatedly executing Python-level instructions.

## 4. Benchmarking Concepts

While studying how to measure execution time, I came across several concepts that affect the reliability of benchmark results.

I learned that `time.time()` provides the current system time, whereas `time.perf_counter_ns()` is designed for measuring short time intervals with high resolution.

I also understood why we should perform multiple measurements instead of trusting a single execution. Execution times can vary because of background processes, system activity, and other sources of measurement variability.

The measurement process itself can also introduce some overhead. This is called measurement overhead, and it is important to consider when measuring very short operations.

I learned that an outlier is a measurement that differs considerably from the rest of the observations. This is one reason why the median is useful when analysing benchmark results.

The median represents the middle observation after sorting the measurements. Unlike the mean, it is less affected by unusually high or low measurements.

I also studied the mode, which represents the most frequently occurring value in a dataset.

## 5. Experimental Observations

Both implementations show approximately linear scaling as the input size increases, which is consistent with the \(O(n)\) time complexity expected for summing a list. However, the built-in `sum()` consistently performs faster than the manual loop. Its speedup also increases with input size, from approximately 3.7× for 100 elements to 7.1× for 1 million elements. This suggests that avoiding repeated Python-level execution overhead becomes increasingly beneficial for larger workloads. Although both methods have the same time complexity, their execution costs differ significantly. The results also show why implementation-level optimizations matter even when the underlying algorithm remains unchanged.

