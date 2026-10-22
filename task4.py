"""Task 4: Static vs dynamic scheduling under irregular workloads."""

import platform
import queue
import threading
import time
from pathlib import Path

STUDENT_ID = 230103328
NUM_ITEMS = 1_000
NUM_WORKERS = 4
DYNAMIC_CHUNK_SIZE = 5
SLEEP_SCALE_SECONDS = 0.0001


def do_work(item: int) -> None:
    """Perform the specified irregular workload; sleep releases the Python GIL."""
    time.sleep(SLEEP_SCALE_SECONDS * (item % 20))


def benchmark(strategy: str) -> tuple[float, float, float, float, list[int]]:
    """Return wall time and worker completion statistics in milliseconds."""
    barrier = threading.Barrier(NUM_WORKERS + 1)
    finished_ns = [0] * NUM_WORKERS
    counts = [0] * NUM_WORKERS

    if strategy == "Static":
        block = NUM_ITEMS // NUM_WORKERS

        def worker(worker_id: int) -> None:
            start = worker_id * block
            end = NUM_ITEMS if worker_id == NUM_WORKERS - 1 else start + block
            barrier.wait()
            for item in range(start, end):
                do_work(item)
                counts[worker_id] += 1
            finished_ns[worker_id] = time.perf_counter_ns()

    elif strategy == "Dynamic":
        chunks: queue.Queue[range] = queue.Queue()
        for start in range(0, NUM_ITEMS, DYNAMIC_CHUNK_SIZE):
            chunks.put(range(start, min(start + DYNAMIC_CHUNK_SIZE, NUM_ITEMS)))

        def worker(worker_id: int) -> None:
            barrier.wait()
            while True:
                try:
                    chunk = chunks.get_nowait()
                except queue.Empty:
                    break
                for item in chunk:
                    do_work(item)
                    counts[worker_id] += 1
            finished_ns[worker_id] = time.perf_counter_ns()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(NUM_WORKERS)]
    for thread in threads:
        thread.start()

    start_ns = time.perf_counter_ns()
    barrier.wait()  # release the four workers together
    for thread in threads:
        thread.join()
    end_ns = time.perf_counter_ns()

    if sum(counts) != NUM_ITEMS:
        raise RuntimeError("Not all 1,000 items were processed")

    first_finish_ms = (min(finished_ns) - start_ns) / 1e6
    last_finish_ms = (max(finished_ns) - start_ns) / 1e6
    completion_delta_ms = (max(finished_ns) - min(finished_ns)) / 1e6
    wall_ms = (end_ns - start_ns) / 1e6
    return wall_ms, first_finish_ms, last_finish_ms, completion_delta_ms, counts


def main() -> None:
    static = benchmark("Static")
    dynamic = benchmark("Dynamic")

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"OS: {platform.platform()}",
        f"Workers (threads): {NUM_WORKERS}",
        f"Items: {NUM_ITEMS} (indices 0 to 999)",
        "Workload: sleep(0.0001 * (i % 20))",
        "Static scheduling: 4 equal contiguous blocks (250 items each)",
        f"Dynamic scheduling: shared work queue, chunk size = {DYNAMIC_CHUNK_SIZE}",
        "Timing: perf_counter_ns() wall-clock, simultaneous thread start",
        "",
        f"{'Strategy':<12} | {'Wall Time':>12} | {'First Finish':>14} | {'Last Finish':>14} | {'Finish Delta':>14}",
        "-" * 79,
    ]
    for name, result in (("Static", static), ("Dynamic", dynamic)):
        wall, first, last, delta, _ = result
        lines.append(
            f"{name:<12} | {wall:>9.3f} ms | {first:>11.3f} ms | {last:>11.3f} ms | {delta:>11.3f} ms"
        )
    lines.extend(
        [
            "",
            f"Static items per worker: {static[4]}",
            f"Dynamic items per worker: {dynamic[4]}",
            f"Static / Dynamic wall-time ratio: {static[0] / dynamic[0]:.3f}x",
            "Note: each completion delta is last worker finish minus first worker finish.",
        ]
    )

    output = "\n".join(lines) + "\n"
    print(output, end="")
    (Path(__file__).resolve().parent / "task4_output.txt").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
