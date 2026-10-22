"""Task 7 — Memory bandwidth saturation vs CPU-compute scaling.
"""

import argparse
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads
from numba.core.config import NUMBA_NUM_THREADS


STUDENT_ID = 230103328
DEFAULT_SIZE = 30_000_000


@njit(parallel=True)
def kernel_a_copy(a, b):
    for i in prange(a.size):
        b[i] = a[i]


@njit(parallel=True)
def kernel_b_trigonometry(a, b):
    for i in prange(a.size):
        x = a[i]
        b[i] = math.sin(x) * math.cos(x) + math.tan(x)


def benchmark(kernel, a, b, workers, repeats):
    """Return median wall-clock kernel runtime in ms, excluding compilation/startup."""
    set_num_threads(workers)
    kernel(a, b)  # compilation / thread startup / page faults not measured
    measurements = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        kernel(a, b)
        measurements.append((time.perf_counter_ns() - start_ns) / 1_000_000)
    return statistics.median(measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        help='Elements in each float64 array (default: 30000000)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Measured runs per configuration (default: 3)')
    args = parser.parse_args()
    if args.size < 1 or args.repeats < 1:
        parser.error('size and repeats must be positive')

    detected_cores = os.cpu_count() or 1
    # Numba's worker pool can be limited by its configuration.
    max_cores = min(detected_cores, NUMBA_NUM_THREADS)
    counts = sorted({1, 2, max_cores} - {n for n in (2,) if n > max_cores})

    a = np.linspace(0.1, 1.0, args.size, dtype=np.float64)
    b = np.empty_like(a)
    lines = []

    def log(message=''):
        print(message)
        lines.append(message)

    log('TASK 7 — MEMORY BANDWIDTH SATURATION')
    log(f'Student ID: {STUDENT_ID}')
    log(f'OS: {platform.platform()}')
    log(f'CPU model/details: {platform.processor() or platform.machine()}')
    log(f'Logical cores detected: {detected_cores}')
    log(f'Numba thread pool limit: {NUMBA_NUM_THREADS}')
    log(f'Workers tested: {counts}')
    log(f'Array size: {args.size:,} float64 elements')
    log(f'Input/output memory: {2 * a.nbytes / (1024**2):.1f} MiB')
    log(f'Timing: median of {args.repeats} wall-clock runs; JIT compilation and warm-up excluded')
    log('')
    log(f"{'Kernel':<21} {'Cores':>5} {'Wall Time':>14} {'Speedup':>10} {'Efficiency':>12}")
    log('-' * 68)

    results = {}
    for label, kernel in (
        ('A — Vector Copy', kernel_a_copy),
        ('B — Trigonometry', kernel_b_trigonometry),
    ):
        measured = {}
        for cores in counts:
            ms = benchmark(kernel, a, b, cores, args.repeats)
            measured[cores] = ms
            speedup = measured[1] / ms
            efficiency = speedup / cores
            log(f'{label:<21} {cores:>5} {ms:>11.3f} ms {speedup:>9.3f}x {efficiency:>11.2%}')
        results[label] = measured

        # One verified result after the benchmark (not part of timings).
        if label.startswith('A'):
            if not np.array_equal(a, b):
                raise AssertionError('Vector copy produced incorrect results')
        else:
            for idx in (0, args.size // 2, args.size - 1):
                x = float(a[idx])
                reference = math.sin(x) * math.cos(x) + math.tan(x)
                if not math.isclose(float(b[idx]), reference, rel_tol=1e-12):
                    raise AssertionError(f'Trigonometry result incorrect at i={idx}')

    copy_times = results['A — Vector Copy']
    compute_times = results['B — Trigonometry']
    log('')
    log('Kernel A effective memory throughput (read A + write B):')
    bandwidth = {}
    for cores, elapsed_ms in copy_times.items():
        gb_per_s = (2 * a.nbytes) / (elapsed_ms / 1000) / 1e9
        bandwidth[cores] = gb_per_s
        log(f'  {cores:>2} workers: {gb_per_s:>8.2f} GB/s')

    log('')
    log('SCALING CONCLUSION')
    log('Memory-channel-saturation candidate: Kernel A (Vector Copy).')
    log('Kernel B (sin/cos/tan) is primarily computation-intensive.')
    if max_cores > 2:
        two = bandwidth[2]
        high = bandwidth[max_cores]
        extra_gain = high / two if two else float('inf')
        ideal_extra_gain = max_cores / 2
        speedup_a = copy_times[1] / copy_times[max_cores]
        speedup_b = compute_times[1] / compute_times[max_cores]
        if extra_gain <= 1.20:
            log(f'Observed: Kernel A throughput barely improved from 2 to {max_cores} workers '
                f'({extra_gain:.2f}x); consistent with memory bandwidth saturation.')
        elif extra_gain < 0.65 * ideal_extra_gain and speedup_a < speedup_b:
            log(f'Observed: Kernel A bandwidth increased only {extra_gain:.2f}x from 2 to '
                f'{max_cores} workers (ideal {ideal_extra_gain:.2f}x), '
                'and scaled worse than Kernel B; consistent with bandwidth pressure.')
        else:
            log('Measured result: no unambiguous bandwidth plateau in this run. '
                'Kernel A remains the memory-bound kernel, but saturation is not established.')
    else:
        log('Only two worker counts available; a scaling plateau cannot be established.')
    log('Note: timings indicate scaling limits, but hardware bandwidth counters are '
        'needed to prove RAM-channel saturation conclusively.')

    output = Path(__file__).with_name('task7_output.txt')
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nSaved: {output}')


if __name__ == '__main__':
    main()
