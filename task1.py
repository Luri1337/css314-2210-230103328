"""Task 1: deterministic serial sum and population standard deviation."""

import math
import os
import platform
import random
import time
from array import array
from pathlib import Path

STUDENT_ID = 230103328
NUM_ELEMENTS = 20_000_000


def cpu_details() -> str:
    """Return the CPU model when available, otherwise platform details."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.startswith("model name"):
                        return line.partition(":")[2].strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().processor or platform.machine()


def main() -> None:
    rng = random.Random(STUDENT_ID)

    # Memory-efficient storage: every item is a 64-bit floating-point number.
    values = array("d", (rng.random() for _ in range(NUM_ELEMENTS)))

    # Purely serial computation; no threads, pools, or vectorized libraries.
    start = time.perf_counter()
    total = math.fsum(values)
    mean = total / NUM_ELEMENTS
    squared_deviations = math.fsum((value - mean) ** 2 for value in values)
    std_dev = math.sqrt(squared_deviations / NUM_ELEMENTS)
    elapsed_ms = (time.perf_counter() - start) * 1000

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"OS: {platform.platform()}",
        f"CPU model/details: {cpu_details()}",
        f"Logical cores (os.cpu_count()): {os.cpu_count()}",
        f"Array size: {NUM_ELEMENTS}",
        "Array dtype: float64 (array('d'))",
        f"Seed: {STUDENT_ID}",
        "Execution mode: serial (single thread)",
        f"Sum: {total:.12f}",
        f"Population standard deviation: {std_dev:.12f}",
        f"Serial computation time: {elapsed_ms:.3f} ms",
    ]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    Path(__file__).with_name("task1_output.txt").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
