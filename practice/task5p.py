"""TASK 5 — Thread Oversubscription Threshold.
"""

import hashlib
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

STUDENT_ID = 230103328
TASK_COUNT = 10_000
HASH_ROUNDS = 500
REPEATS = 3
CORES = max(1, os.cpu_count() or 1)

# Every task hashes exactly the same 1 KiB input but uses its own salt.
SOURCE = str(STUDENT_ID).encode("ascii")
PAYLOAD = (SOURCE * (1024 // len(SOURCE) + 1))[:1024]


def work(item_index: int) -> int:
    # PBKDF2-HMAC performs 500 SHA-256-based rounds in OpenSSL native code.
    # Unlike a Python loop over 1 KiB hashlib calls, this can release the GIL,
    # allowing real multi-core activity in CPython.
    salt = item_index.to_bytes(8, byteorder="little", signed=False)
    digest = hashlib.pbkdf2_hmac("sha256", PAYLOAD, salt, HASH_ROUNDS)
    return digest[0]


def start_workers(pool: ThreadPoolExecutor, count: int) -> None:
    """Warm up and create all requested OS threads before taking measurements."""
    gate = Barrier(count + 1, timeout=60)
    futures = [pool.submit(gate.wait) for _ in range(count)]
    gate.wait()
    for future in futures:
        future.result()


def pool_sizes(cores: int) -> list[int]:
    # Deduplicate small-core cases (e.g. Cores/2 = 1).
    candidates = [1, max(1, cores // 2), cores, cores * 2, cores * 4,
                  cores * 16, 256]
    return list(dict.fromkeys(candidates))


def benchmark(count: int) -> tuple[float, int]:
    runs_ms = []
    check_sum = None
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="bench") as pool:
        start_workers(pool, count)
        for _ in range(REPEATS):
            started = time.perf_counter()
            # Executor overhead / submission / waiting is included in the clock.
            result_sum = sum(pool.map(work, range(TASK_COUNT)))
            elapsed_ms = (time.perf_counter() - started) * 1000
            runs_ms.append(elapsed_ms)
            if check_sum is None:
                check_sum = result_sum
            elif check_sum != result_sum:
                raise RuntimeError("Results were not consistent between repeats")
    return statistics.median(runs_ms), check_sum


def main() -> None:
    rows = []
    ref_sum = None
    for count in pool_sizes(CORES):
        elapsed_ms, total = benchmark(count)
        if ref_sum is None:
            ref_sum = total
        elif ref_sum != total:
            raise RuntimeError("Inconsistent checksums across pool sizes")
        rows.append((count, elapsed_ms))

    baseline = rows[0][1]
    # The first strict measured regression relative to the previous tested size.
    first_regression = next((i for i in range(1, len(rows))
                             if rows[i][1] > rows[i - 1][1]), None)
    # First oversubscribed size that is slower than the best at or below Cores.
    best_not_oversubscribed = min(t for p, t in rows if p <= CORES)
    first_oversub_regression = next((i for i, (p, t) in enumerate(rows)
                                    if p > CORES and t > best_not_oversubscribed), None)

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"OS: {platform.platform()}",
        f"Logical cores (os.cpu_count): {CORES}",
        f"Total tasks per run: {TASK_COUNT:,}",
        f"CPU work per task: PBKDF2-HMAC-SHA256 (500 rounds, 1 KiB input)",
        f"Repetitions per pool size: {REPEATS} (median wall time)",
        "Worker threads prestarted; 10,000 task submissions/completions timed.",
        "",
        "Pool Size | Execution Time (ms) | Speedup vs 1 thread | Status",
        "-" * 79,
    ]
    for i, (p, t) in enumerate(rows):
        status = []
        if p > CORES:
            status.append("OVERSUBSCRIBED")
        if i == first_regression:
            status.append("FIRST SLOWDOWN")
        if i == first_oversub_regression:
            status.append("OVERSUB THRESHOLD")
        if not status:
            status = ["BASELINE" if p == 1 else "OK"]
        lines.append(f"{p:9d} | {t:19.3f} | {baseline/t:18.3f}x | {', '.join(status)}")

    lines += ["", f"Result checksum (same for all runs): {ref_sum}"]
    if first_regression is None:
        lines.append("First measured slowdown: not observed among tested pool sizes.")
    else:
        i = first_regression
        lines.append(f"First measured slowdown: {rows[i][0]} threads "
                     f"({rows[i][1]:.3f} ms vs {rows[i-1][1]:.3f} ms "
                     f"at {rows[i-1][0]} threads).")
    if first_oversub_regression is None:
        lines.append("Oversubscription degradation threshold: not observed in tested sizes.")
    else:
        i = first_oversub_regression
        lines.append(f"First measured oversubscription degradation: {rows[i][0]} threads "
                     f"({rows[i][1]:.3f} ms vs best <= {CORES} cores: "
                     f"{best_not_oversubscribed:.3f} ms).")
    lines.append("Note: wall-clock slowdown indicates contention/scheduling overhead; "
                 "it does not alone prove OS context-switching as the sole cause.")

    output = "\n".join(lines) + "\n"
    print(output, end="")
    Path(__file__).with_name("task5_output.txt").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
