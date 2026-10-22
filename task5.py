"""TASK 5 — Parallel overhead crossover for y[i] = a*x[i] + b.
"""

import os
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from numba import njit

STUDENT_ID = 230103328
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
REPEATS = 9
A = 2.5
B = 1.75
OUTPUT = Path(__file__).with_name("task5_output.txt")


@njit(nogil=True)
def affine_kernel(x, y, a, b, start, end):
    """Write only [start:end); all worker segments are disjoint."""
    for i in range(start, end):
        y[i] = a * x[i] + b


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_parallel(pool, x, y, worker_count):
    n = x.size
    futures = [
        pool.submit(
            affine_kernel, x, y, A, B,
            worker * n // worker_count,
            (worker + 1) * n // worker_count,
        )
        for worker in range(worker_count)
    ]
    for future in futures:
        future.result()  # Wait for every worker, propagating exceptions.


def measure(operation):
    start = time.perf_counter_ns()
    operation()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def benchmark(pool, n, workers):
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)

    def serial():
        affine_kernel(x, y, A, B, 0, n)

    def parallel():
        run_parallel(pool, x, y, workers)

    # Warm caches / memory pages before measuring; JIT was compiled earlier.
    serial()
    parallel()

    serial_times = []
    parallel_times = []
    for repetition in range(REPEATS):
        # Alternate order to reduce systematic CPU-frequency/cache bias.
        if repetition % 2 == 0:
            serial_times.append(measure(serial))
            parallel_times.append(measure(parallel))
        else:
            parallel_times.append(measure(parallel))
            serial_times.append(measure(serial))

    # Verify correctness outside the measured interval.
    step = max(1, n // 1_000)
    if not np.allclose(y[::step], A * x[::step] + B, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"Incorrect result for N={n}")

    return statistics.median(serial_times), statistics.median(parallel_times)


def main():
    workers = os.cpu_count() or 1
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical cores / worker threads: {workers}")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print("Array dtype: float64")
    print(f"Array sizes: {SIZES}")
    print(f"Repetitions per method: {REPEATS} (median wall-clock time)")
    print("Timing: computation + thread submission/join; excludes array allocation,")
    print("        Numba compilation and initial worker-thread startup")
    print("Execution: serial Numba vs. {0} Numba nogil worker threads".format(workers))
    print()

    # Compile before timing (same machine-code kernel for either method).
    warm_x = np.array([0.0, 1.0], dtype=np.float64)
    warm_y = np.empty_like(warm_x)
    affine_kernel(warm_x, warm_y, A, B, 0, 2)

    results = []
    print(f"{'N':>12} | {'Serial (ms)':>14} | {'Parallel (ms)':>14} | {'Speedup':>9} | Result")
    print("-" * 77)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Ensure all threads exist BEFORE the timed experiments.
        ready = threading.Barrier(workers + 1)
        warmups = [pool.submit(ready.wait) for _ in range(workers)]
        ready.wait()
        for future in warmups:
            future.result()

        for n in SIZES:
            serial_ms, parallel_ms = benchmark(pool, n, workers)
            speedup = serial_ms / parallel_ms
            faster = parallel_ms < serial_ms
            status = "PARALLEL FASTER" if faster else "PARALLEL SLOWER/EQUAL"
            results.append((n, serial_ms, parallel_ms, faster))
            print(
                f"{n:12,d} | {serial_ms:14.6f} | {parallel_ms:14.6f} "
                f"| {speedup:8.3f}x | {status}"
            )

    print()
    print("CROSSOVER ANALYSIS (based on measured array sizes):")
    transitions = []
    for previous, current in zip(results, results[1:]):
        if not previous[3] and current[3]:
            transitions.append((previous[0], current[0]))

    if transitions:
        for low, high in transitions:
            print(f"SLOWER -> FASTER between N={low:,} and N={high:,}")
            print(f"Observed crossover range: ({low:,}, {high:,}]")
        if len(transitions) > 1:
            print("Multiple transitions observed; scheduling/memory effects may be non-monotonic.")
    else:
        if all(not row[3] for row in results):
            print("No crossover in tested range: parallel never beat serial.")
        elif all(row[3] for row in results):
            print("No crossover in tested range: parallel was faster even at N=100.")
        else:
            print("No SLOWER -> FASTER transition in ascending tested sizes.")

    print("Note: this is the bracket between measured sizes, not an exact single")
    print("element threshold; timings can change across runs and CPU load.")


if __name__ == "__main__":
    with OUTPUT.open("w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            main()
