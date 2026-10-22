"""Task 6: Race condition forensics with four real Python threads
"""

from pathlib import Path
import os
import platform
import threading
import time

STUDENT_ID = 230103328
THREAD_COUNT = 4
INCREMENTS_PER_THREAD = 2_000_000
YIELD_INTERVAL = 10_000
EXPECTED = THREAD_COUNT * INCREMENTS_PER_THREAD


def run_trial(use_lock: bool) -> tuple[int, float]:
    shared_counter = [0]  # A mutable container shared by all four threads.
    lock = threading.Lock()

    def worker() -> None:
        if use_lock:
            for i in range(INCREMENTS_PER_THREAD):
                # Protect the entire read-modify-write sequence.
                with lock:
                    old_value = shared_counter[0]
                    if i % YIELD_INTERVAL == 0:
                        time.sleep(0)  # Deliberate scheduling point in BOTH tests.
                    shared_counter[0] = old_value + 1
        else:
            for i in range(INCREMENTS_PER_THREAD):
                # Intentionally unsafe: another thread can update the value
                # after it is read but before this thread writes it back.
                old_value = shared_counter[0]
                if i % YIELD_INTERVAL == 0:
                    time.sleep(0)
                shared_counter[0] = old_value + 1

    workers = [threading.Thread(target=worker, name=f"worker-{i}")
               for i in range(THREAD_COUNT)]

    start = time.perf_counter()
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return shared_counter[0], elapsed_ms


def main() -> None:
    unsafe_count, unsafe_ms = run_trial(use_lock=False)
    safe_count, safe_ms = run_trial(use_lock=True)

    lost = EXPECTED - unsafe_count
    lost_pct = 100.0 * lost / EXPECTED
    penalty_ms = safe_ms - unsafe_ms
    penalty_pct = 100.0 * penalty_ms / unsafe_ms
    penalty_ratio = safe_ms / unsafe_ms

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"OS: {platform.platform()}",
        f"CPU model/details: {platform.processor() or platform.machine()}",
        f"Logical cores: {os.cpu_count()}",
        f"Parallel threads: {THREAD_COUNT}",
        f"Increments per thread: {INCREMENTS_PER_THREAD:,}",
        f"Expected final value: {EXPECTED:,}",
        f"Yield interval: {YIELD_INTERVAL:,} updates (after read, before write)",
        "",
        "Mode                   | Actual Count | Lost Updates | Lost %    | Wall Time",
        "--------------------------------------------------------------------------",
        f"Unprotected (no lock)  | {unsafe_count:12,} | {lost:12,} | {lost_pct:8.3f}% | {unsafe_ms:9.3f} ms",
        f"Protected (mutex lock) | {safe_count:12,} | {EXPECTED - safe_count:12,} | {100*(EXPECTED-safe_count)/EXPECTED:8.3f}% | {safe_ms:9.3f} ms",
        "",
        f"Lock time penalty: {penalty_ms:+.3f} ms ({penalty_pct:+.2f}%)",
        f"Lock/unprotected time ratio: {penalty_ratio:.3f}x",
        f"Protected result correct: {safe_count == EXPECTED}",
    ]

    output = "\n".join(lines) + "\n"
    print(output, end="")
    destination = Path(__file__).with_name("task6_output.txt")
    destination.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
