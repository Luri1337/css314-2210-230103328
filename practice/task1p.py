"""Task 1: Row-major versus column-major cache traversal.
"""

import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
from numba import njit

STUDENT_ID = 230103328
SIDE = 8_000
REPEATS = 3


@njit(nogil=True)
def row_major_sum(matrix, begin, end):
    acc = 0.0
    for row in range(begin, end):
        for col in range(matrix.shape[1]):
            acc += matrix[row, col]
    return acc


@njit(nogil=True)
def column_major_sum(matrix, begin, end):
    acc = 0.0
    for col in range(begin, end):
        for row in range(matrix.shape[0]):
            acc += matrix[row, col]
    return acc


def split_ranges(length, workers):
    return [(i * length // workers, (i + 1) * length // workers)
            for i in range(workers)]


def benchmark(pool, function, matrix, ranges):
    start = time.perf_counter_ns()
    futures = [pool.submit(function, matrix, begin, end)
               for begin, end in ranges]
    result = sum(future.result() for future in futures)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    return elapsed_ms, result


def main():
    output = Path(__file__).with_name('task1_output.txt')
    with output.open('w', encoding='utf-8') as output_file:
        def log(line=''):
            print(line, flush=True)
            print(line, file=output_file, flush=True)

        logical = os.cpu_count() or 1
        physical = psutil.cpu_count(logical=False) or logical
        workers = max(1, physical)

        log(f'Student ID: {STUDENT_ID}')
        log(f'OS: {platform.platform()}')
        log(f'Logical cores: {logical}')
        log(f'Physical cores / worker threads: {workers}')
        log(f'Matrix: {SIDE} x {SIDE}, dtype=float64, C-contiguous')
        log(f'Seed: {STUDENT_ID}')
        log(f'Timing: median of {REPEATS} runs, including task scheduling; JIT warm-up excluded')

        # One C-contiguous allocation (~512 MB); NumPy fills it with the fixed seed.
        matrix = np.random.default_rng(STUDENT_ID).random((SIDE, SIDE), dtype=np.float64)
        warm = matrix[:8, :8].copy()
        row_major_sum(warm, 0, 8)  # JIT-compile outside measurements.
        column_major_sum(warm, 0, 8)

        ranges = split_ranges(SIDE, workers)
        samples = {'A': [], 'B': []}
        checksums = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Alternate starting mode so that the first run has no fixed advantage.
            for repetition in range(REPEATS):
                order = ('A', 'B') if repetition % 2 == 0 else ('B', 'A')
                for mode in order:
                    fn = row_major_sum if mode == 'A' else column_major_sum
                    ms, total = benchmark(pool, fn, matrix, ranges)
                    samples[mode].append(ms)
                    checksums[mode] = total

        if not np.isclose(checksums['A'], checksums['B'], rtol=1e-10, atol=1e-6):
            log('Verification: FAIL (row and column sums disagree)')
            raise SystemExit(1)

        a_ms = statistics.median(samples['A'])
        b_ms = statistics.median(samples['B'])
        log(f'Row-major sum: {checksums["A"]:.9f}')
        log(f'Column-major sum: {checksums["B"]:.9f}')
        log('Verification: PASS')
        log(f'Mode A (Row-Major):       {a_ms:,.3f} ms')
        log(f'Mode B (Column-Major):    {b_ms:,.3f} ms')
        log(f'Cache penalty B / A:     {b_ms / a_ms:.3f}x')


if __name__ == '__main__':
    main()
