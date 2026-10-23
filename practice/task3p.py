"""Task 3: Serial versus two-pass parallel inclusive prefix scan.
"""

import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
from numba import njit

STUDENT_ID = 230103328
N = 2_000_000


@njit(nogil=True)
def serial_inclusive_scan(source, destination):
    running = 0
    for i in range(source.size):
        running += source[i]
        destination[i] = running


@njit(nogil=True)
def local_inclusive_scan(source, destination, begin, end):
    running = 0
    for i in range(begin, end):
        running += source[i]
        destination[i] = running
    return running


@njit(nogil=True)
def apply_offset(destination, begin, end, offset):
    for i in range(begin, end):
        destination[i] += offset


def parallel_inclusive_scan(pool, source, destination, ranges):
    # Pass 1: Each thread builds an inclusive prefix inside its own chunk.
    futures = [pool.submit(local_inclusive_scan, source, destination, begin, end)
               for begin, end in ranges]
    chunk_sums = [f.result() for f in futures]

    # Small ordered prefix reduction over chunk totals.
    offsets = []
    running = 0
    for chunk_sum in chunk_sums:
        offsets.append(running)
        running += chunk_sum

    # Pass 2: Add the preceding chunks' totals to each local prefix.
    futures = [pool.submit(apply_offset, destination, begin, end, int(offset))
               for (begin, end), offset in zip(ranges, offsets)]
    for future in futures:
        future.result()


def main():
    with Path(__file__).with_name('task3_output.txt').open('w', encoding='utf-8') as out:
        def log(line=''):
            print(line, flush=True)
            print(line, file=out, flush=True)

        logical = os.cpu_count() or 1
        workers = max(1, min(psutil.cpu_count(logical=False) or logical, N))
        data = np.random.default_rng(STUDENT_ID).integers(1, 11, size=N, dtype=np.int64)
        serial_result = np.empty(N, dtype=np.int64)
        parallel_result = np.empty(N, dtype=np.int64)

        # Compile ahead of time; JIT time is not part of the benchmark.
        small = np.array([1, 2, 3], dtype=np.int64)
        warm = np.empty_like(small)
        serial_inclusive_scan(small, warm)
        local_inclusive_scan(small, warm, 0, 3)
        apply_offset(warm, 0, 3, 0)

        log(f'Student ID: {STUDENT_ID}')
        log(f'OS: {platform.platform()}')
        log(f'Logical cores: {logical}')
        log(f'Physical core workers: {workers}')
        log(f'Input elements: {N:,}, integers in [1, 10]')
        log(f'Seed: {STUDENT_ID}')
        log('Timing: scan only, excluding input creation / JIT compilation / output allocation')

        ranges = [(i * N // workers, (i + 1) * N // workers)
                  for i in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            start = time.perf_counter_ns()
            serial_inclusive_scan(data, serial_result)
            serial_ms = (time.perf_counter_ns() - start) / 1_000_000

            start = time.perf_counter_ns()
            parallel_inclusive_scan(pool, data, parallel_result, ranges)
            parallel_ms = (time.perf_counter_ns() - start) / 1_000_000

        passed = (parallel_result[-1] == serial_result[-1]) and np.array_equal(
            parallel_result, serial_result
        )
        log('')
        log('Serial Time (ms) | Parallel Time (ms) | Speedup | Verification')
        log('---------------------------------------------------------------')
        log(f'{serial_ms:16.3f} | {parallel_ms:18.3f} | {serial_ms / parallel_ms:7.3f}x | {"PASS" if passed else "FAIL"}')
        log(f'Serial final prefix: {serial_result[-1]}')
        log(f'Parallel final prefix: {parallel_result[-1]}')
        if not passed:
            raise SystemExit(1)
        # Required explicit final-value assertion (and full-array check above).
        assert parallel_result[-1] == serial_result[-1]


if __name__ == '__main__':
    main()
