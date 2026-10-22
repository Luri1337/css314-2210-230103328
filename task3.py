"""Task 3: False sharing profile (real parallel threads via Numba, GIL released).
"""

import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

STUDENT_ID = 230103328
INCREMENTS = 50_000_000
CACHE_LINE_BYTES = 64
INT_BYTES = np.dtype(np.int64).itemsize
STRIDE = CACHE_LINE_BYTES // INT_BYTES  # 8 int64 values = 64 bytes


def physical_core_count():
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        if cores and cores > 0:
            return cores
    except ImportError:
        pass

    if sys.platform == "win32":
        try:
            command = (
                "(Get-CimInstance Win32_Processor | "
                "Measure-Object -Property NumberOfCores -Sum).Sum"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, text=True, timeout=15, check=True,
            )
            result = int(completed.stdout.strip())
            if result > 0:
                return result
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True
            ).strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    raise RuntimeError(
        "Could not detect physical cores. Install psutil: python -m pip install psutil"
    )


@intrinsic
def compiler_barrier(typing_context):
    """Stop LLVM from collapsing 50M updates into just one array write.

    Empty compiler barrier, NOT a hardware memory fence: no extra CPU operation.
    The memory clobber forces each iteration to actually load/store the counter.
    """
    signature = types.void()

    def codegen(context, builder, signature, args):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), []),
            "", "~{memory}", side_effect=True,
        )
        builder.call(asm, [])
        return context.get_dummy_value()

    return signature, codegen


@njit(nogil=True)
def increment_counter(counts, index, increments):
    # Compiled native loop with the Python GIL released: genuine parallel threads.
    for _ in range(increments):
        counts[index] += 1
        compiler_barrier()


def benchmark(n_threads, padded):
    if padded:
        shared_counts = np.zeros(n_threads * STRIDE, dtype=np.int64)
        indexes = [i * STRIDE for i in range(n_threads)]
    else:
        shared_counts = np.zeros(n_threads, dtype=np.int64)
        indexes = list(range(n_threads))

    gate = threading.Barrier(n_threads + 1)
    errors = []

    def work(index):
        try:
            gate.wait()
            increment_counter(shared_counts, index, INCREMENTS)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(index,)) for index in indexes]
    for thread in threads:
        thread.start()

    start_ns = time.perf_counter_ns()
    gate.wait()
    for thread in threads:
        thread.join()
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

    if errors:
        raise RuntimeError("Worker thread failed") from errors[0]
    if any(int(shared_counts[index]) != INCREMENTS for index in indexes):
        raise AssertionError("Incorrect counter result")

    return elapsed_ms


def main():
    n_threads = physical_core_count()

    # Warm up JIT compilation OUTSIDE benchmarks.
    warmup = np.zeros(1, dtype=np.int64)
    increment_counter(warmup, 0, 1)

    unpadded_ms = benchmark(n_threads, padded=False)
    padded_ms = benchmark(n_threads, padded=True)
    ratio = unpadded_ms / padded_ms

    lines = [
        f"Student ID: {STUDENT_ID}",
        f"OS: {platform.platform()}",
        f"CPU model/details: {platform.processor() or platform.machine()}",
        f"Logical cores: {os.cpu_count()}",
        f"Physical cores / threads: {n_threads}",
        f"Increments per thread: {INCREMENTS}",
        "Execution mode: native multithreading (Numba nogil=True)",
        "Counter type: int64",
        "Unpadded spacing: 8 bytes",
        "Padded spacing: 64 bytes",
        "JIT compilation and worker creation excluded from timing",
        "",
        "Variant            |      Wall Time",
        "-------------------------------------",
        f"Unpadded Shared    | {unpadded_ms:12.3f} ms",
        f"Padded Shared      | {padded_ms:12.3f} ms",
        "",
        f"Unpadded / Padded ratio: {ratio:.3f}x",
        f"Final count per thread (both variants): {INCREMENTS}",
        f"Total increments per variant: {n_threads * INCREMENTS}",
    ]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    output_path = Path(__file__).with_name("task3_output.txt")
    output_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
