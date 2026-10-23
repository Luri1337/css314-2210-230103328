"""Task 2: Monte Carlo pi with locked shared PRNG vs independent PRNGs.
"""

import math
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

STUDENT_ID = 230103328
SAMPLES = 40_000_000
WORKERS = 4
BATCH_SIZE = 100_000


def count_hits(generator, count, lock=None):
    inside = 0
    for start in range(0, count, BATCH_SIZE):
        n = min(BATCH_SIZE, count - start)
        if lock is None:
            x = generator.random(n)
            y = generator.random(n)
        else:
            # The same PRNG state can only be touched by one thread at a time.
            with lock:
                x = generator.random(n)
                y = generator.random(n)
        inside += int(np.count_nonzero(x * x + y * y <= 1.0))
    return inside


def run(shared):
    counts = [SAMPLES // WORKERS] * WORKERS
    shared_generator = np.random.default_rng(STUDENT_ID) if shared else None
    generator_lock = threading.Lock() if shared else None
    start = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = []
        for i, count in enumerate(counts):
            generator = (shared_generator if shared else
                         np.random.default_rng((STUDENT_ID * 10) + i))
            futures.append(pool.submit(count_hits, generator, count, generator_lock))
        inside = sum(future.result() for future in futures)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    pi_estimate = 4.0 * inside / SAMPLES
    return pi_estimate, abs(pi_estimate - math.pi), elapsed_ms


def main():
    with Path(__file__).with_name('task2_output.txt').open('w', encoding='utf-8') as out:
        def log(line=''):
            print(line, flush=True)
            print(line, file=out, flush=True)

        log(f'Student ID: {STUDENT_ID}')
        log(f'OS: {platform.platform()}')
        log(f'Total Monte Carlo samples per run: {SAMPLES:,}')
        log(f'Worker threads: {WORKERS}')
        log(f'Batch size: {BATCH_SIZE:,} points')
        log(f'Shared seed: {STUDENT_ID}')
        log(f'Independent seeds: {[(STUDENT_ID * 10) + i for i in range(WORKERS)]}')
        log('Both runs include thread startup and counting; the shared generator is mutex-protected.')

        shared_pi, shared_err, shared_ms = run(shared=True)
        local_pi, local_err, local_ms = run(shared=False)

        log('')
        log('Mode                Estimated pi   Abs. error         Wall time')
        log('-------------------------------------------------------------------')
        log(f'Shared PRNG + mutex  {shared_pi:.9f}   {shared_err:.9f}      {shared_ms:,.3f} ms')
        log(f'Independent PRNGs    {local_pi:.9f}   {local_err:.9f}      {local_ms:,.3f} ms')
        log(f'Lock elimination speedup (shared / independent): {shared_ms / local_ms:.3f}x')
        if local_ms >= shared_ms:
            log('Note: Lock elimination did not improve wall time on this particular run.')


if __name__ == '__main__':
    main()
