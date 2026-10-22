"""Task 2: benchmark shared-memory, multi-process float64 summation.
"""

import math
import multiprocessing as mp
import os
import platform
import random
import time
from multiprocessing import shared_memory
from pathlib import Path

STUDENT_ID = 230103328
ARRAY_SIZE = 20_000_000
FLOAT64_BYTES = 8
PARALLEL_FRACTION = 0.95


def sum_chunk(job: tuple[str, int, int]) -> float:
    """Sum a disjoint range from the shared float64 array in a worker process."""
    shm_name, start_index, end_index = job
    block = shared_memory.SharedMemory(name=shm_name)
    values = block.buf.cast("d")
    try:
        return math.fsum(values[start_index:end_index])
    finally:
        values.release()
        block.close()


def signal_ready(ready_queue: mp.Queue) -> None:
    """Let the parent exclude worker startup from the timed summation."""
    ready_queue.put(os.getpid())


def amdahl_bound(workers: int) -> float:
    return 1.0 / ((1.0 - PARALLEL_FRACTION) + PARALLEL_FRACTION / workers)


def benchmark(ctx: mp.context.BaseContext, shm_name: str, workers: int) -> tuple[float, float]:
    # Partition the full array; each value is read exactly once.
    jobs = [
        (shm_name, ARRAY_SIZE * i // workers, ARRAY_SIZE * (i + 1) // workers)
        for i in range(workers)
    ]

    # Spawn and initialize the worker processes BEFORE starting the timer.
    ready_queue = ctx.Queue()
    try:
        with ctx.Pool(
            processes=workers,
            initializer=signal_ready,
            initargs=(ready_queue,),
        ) as pool:
            for _ in range(workers):
                ready_queue.get(timeout=120)

            start_ns = time.perf_counter_ns()
            partial_sums = pool.map(sum_chunk, jobs, chunksize=1)
            result = math.fsum(partial_sums)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    finally:
        ready_queue.close()
        ready_queue.join_thread()

    return result, elapsed_ms


def main() -> None:
    cores = os.cpu_count() or 1
    worker_counts = list(dict.fromkeys([1, 2, 4, cores, cores * 2]))

    # Populate the same seeded float64 values as in Task 1.
    block = shared_memory.SharedMemory(create=True, size=ARRAY_SIZE * FLOAT64_BYTES)
    try:
        values = block.buf.cast("d")
        try:
            rng = random.Random(STUDENT_ID)
            for i in range(ARRAY_SIZE):
                values[i] = rng.random()
        finally:
            values.release()

        context = mp.get_context("spawn")  # Portable on Windows, macOS and Linux.
        measurements = []
        baseline_sum = None
        baseline_ms = None

        for workers in worker_counts:
            total, elapsed_ms = benchmark(context, block.name, workers)
            if baseline_sum is None:
                baseline_sum = total
                baseline_ms = elapsed_ms
            elif not math.isclose(total, baseline_sum, rel_tol=1e-13, abs_tol=1e-9):
                raise RuntimeError(f"Sum differs for {workers} workers: {total} vs {baseline_sum}")
            measurements.append((workers, elapsed_ms, baseline_ms / elapsed_ms, amdahl_bound(workers)))

        lines = [
            f"Student ID: {STUDENT_ID}",
            f"OS: {platform.platform()}",
            f"CPU model/details: {platform.processor() or platform.machine()}",
            f"Logical cores: {cores}",
            f"Array size: {ARRAY_SIZE}",
            "Array dtype: float64 (shared memory)",
            f"Seed: {STUDENT_ID}",
            "Execution mode: multiprocessing (one worker process per count)",
            "Timing: wall-clock summation + partial-sum reduction; worker startup excluded",
            f"Reference sum: {baseline_sum:.12f}",
            f"Parallel fraction P: {PARALLEL_FRACTION:.2f}",
            "",
            f"{'Cores':>5} | {'Wall Time':>14} | {'Speedup':>9} | {'Amdahl Upper Bound':>18}",
            "-" * 59,
        ]
        for workers, elapsed_ms, speedup, bound in measurements:
            lines.append(f"{workers:>5} | {elapsed_ms:>10.3f} ms | {speedup:>8.3f}x | {bound:>17.3f}x")

        output = "\n".join(lines) + "\n"
        print(output, end="")
        Path(__file__).with_name("task2_output.txt").write_text(output, encoding="utf-8")
    finally:
        block.close()
        block.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
