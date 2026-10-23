"""Task 4 (CPU route): Pickle/Pipe IPC versus zero-copy shared-memory transfer.
Install: python -m pip install numpy
Run:     python task4.py
Creates: task4_output.txt next to this file.
"""

import multiprocessing as mp
import platform
import statistics
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

STUDENT_ID = 230103328
SIZES = (100_000, 1_000_000, 10_000_000)
REPEATS = 3


def pickle_receiver(conn):
    try:
        while True:
            message = conn.recv()
            if message is None:
                return
            # Parent sends the entire array, copied via pickle and the OS pipe.
            array = message
            conn.send((len(array), float(array[0]), float(array[-1])))
    finally:
        conn.close()


def shared_memory_receiver(conn):
    try:
        while True:
            message = conn.recv()
            if message is None:
                return
            shm_name, n = message
            segment = shared_memory.SharedMemory(name=shm_name)
            try:
                # Attach to already populated shared bytes: no payload copy.
                array = np.ndarray((n,), dtype=np.float64, buffer=segment.buf)
                result = (len(array), float(array[0]), float(array[-1]))
                del array  # Release the memoryview before closing on Windows.
            finally:
                segment.close()
            conn.send(result)
    finally:
        conn.close()


def measure_pipe(connection, array):
    t0 = time.perf_counter_ns()
    connection.send(array)  # This serializes/transfers the whole array.
    received = connection.recv()  # Round-trip acknowledgment, same in both paths.
    elapsed = (time.perf_counter_ns() - t0) / 1_000_000
    return elapsed, received


def measure_shm(connection, name, n):
    t0 = time.perf_counter_ns()
    connection.send((name, n))  # Only small metadata goes through the pipe.
    received = connection.recv()
    elapsed = (time.perf_counter_ns() - t0) / 1_000_000
    return elapsed, received


def main():
    with Path(__file__).with_name('task4_output.txt').open('w', encoding='utf-8') as out:
        def log(line=''):
            print(line, flush=True)
            print(line, file=out, flush=True)

        log(f'Student ID: {STUDENT_ID}')
        log(f'OS: {platform.platform()}')
        log(f'CPU IPC benchmark, NumPy float64, sizes: {list(SIZES)}')
        log(f'Median of {REPEATS} transfers per size; worker startup excluded')
        log('Path A: send full array through multiprocessing.Pipe (Pickle + data copy)')
        log('Path B: send shared_memory name and shape (zero-copy data access)')
        log('Times include parent send, worker receive/attach, and small acknowledgment.')
        log('Shared-memory allocation and initial array filling excluded from both paths.')

        context = mp.get_context('spawn')  # Portable to Windows.
        segment = shared_memory.SharedMemory(create=True, size=max(SIZES) * 8)
        p_conn, child_conn = context.Pipe(duplex=True)
        s_conn, s_child_conn = context.Pipe(duplex=True)
        p_proc = context.Process(target=pickle_receiver, args=(child_conn,))
        s_proc = context.Process(target=shared_memory_receiver, args=(s_child_conn,))
        try:
            full = np.ndarray((max(SIZES),), dtype=np.float64, buffer=segment.buf)
            np.random.default_rng(STUDENT_ID).random(max(SIZES), out=full)
            p_proc.start()
            s_proc.start()
            child_conn.close()
            s_child_conn.close()

            log('')
            log('         N | Path A: Pipe (ms) | Path B: Shared (ms) | A / B')
            log('----------------------------------------------------------')
            for n in SIZES:
                array = full[:n]  # Identical source bytes for both comparisons.
                expected = (n, float(array[0]), float(array[-1]))
                pipe_samples = []
                shm_samples = []
                for trial in range(REPEATS):
                    # Alternate order to reduce temperature / contention bias.
                    if trial % 2 == 0:
                        a_ms, a_value = measure_pipe(p_conn, array)
                        b_ms, b_value = measure_shm(s_conn, segment.name, n)
                    else:
                        b_ms, b_value = measure_shm(s_conn, segment.name, n)
                        a_ms, a_value = measure_pipe(p_conn, array)
                    if a_value != expected or b_value != expected:
                        raise RuntimeError('IPC data verification failed')
                    pipe_samples.append(a_ms)
                    shm_samples.append(b_ms)
                a_median = statistics.median(pipe_samples)
                b_median = statistics.median(shm_samples)
                log(f'{n:10,d} | {a_median:17,.3f} | {b_median:19,.3f} | {a_median / b_median:6.2f}x')
            log('Verification: PASS (matching length and sampled values)')
        finally:
            if p_proc.is_alive():
                p_conn.send(None)
            if s_proc.is_alive():
                s_conn.send(None)
            p_conn.close()
            s_conn.close()
            child_conn.close()
            s_child_conn.close()
            if p_proc.pid is not None:
                p_proc.join(timeout=10)
                if p_proc.is_alive():
                    p_proc.terminate()
                    p_proc.join()
            if s_proc.pid is not None:
                s_proc.join(timeout=10)
                if s_proc.is_alive():
                    s_proc.terminate()
                    s_proc.join()
            if 'array' in locals():
                del array
            del full
            segment.close()
            segment.unlink()


if __name__ == '__main__':
    mp.freeze_support()
    main()
