from __future__ import annotations

import os
import queue
import random
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator

import numpy as np
import multiprocessing as mp


class IPCSharedBuffer:
    """Cross-version shared buffer for numpy batch exchange across workers."""

    def __init__(self, size: int | None = None, name: str | None = None):
        if size is None and name is None:
            raise ValueError("Provide size to create a buffer or name to attach one")

        self._is_shm = False
        self._shm = None
        self._fd: int | None = None
        self._path: str | None = None
        self._mmap = None

        if os.sys.version_info >= (3, 8):
            from multiprocessing import shared_memory

            if name is None:
                self._shm = shared_memory.SharedMemory(create=True, size=int(size or 0))
            else:
                self._shm = shared_memory.SharedMemory(name=name)
            self._is_shm = True
            self._buffer = self._shm.buf
            self.size = int(self._shm.size)
            return

        tmpdir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
        if name is None:
            fd, path = tempfile.mkstemp(dir=tmpdir)
            os.ftruncate(fd, int(size or 0))
        else:
            path = name
            fd = os.open(path, os.O_RDWR)

        import mmap

        self._fd = fd
        self._path = path
        self.size = os.stat(fd).st_size
        self._mmap = mmap.mmap(fd, self.size)
        self._buffer = memoryview(self._mmap)

    @property
    def buf(self) -> memoryview:
        return self._buffer

    def name_or_path(self) -> str:
        if self._is_shm:
            return str(self._shm.name)
        assert self._path is not None
        return self._path

    def close(self) -> None:
        if self._is_shm:
            if self._shm is not None:
                self._shm.close()
            return

        if self._buffer is not None:
            self._buffer.release()
        if self._mmap is not None:
            self._mmap.close()
        if self._fd is not None:
            os.close(self._fd)

    def unlink(self) -> None:
        if self._is_shm:
            if self._shm is not None:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            return

        if self._path:
            try:
                os.unlink(self._path)
            except FileNotFoundError:
                pass


def _worker_loop(
    data_chunk: list[Any],
    read_func: Callable[[Any], Any],
    ipc_name: str,
    buffer_queue: mp.Queue,
    ready_queue: mp.Queue,
    batch_size: int,
    input_dtype: np.dtype,
    qsize: int,
    no_of_worker_threads: int,
    infinite: bool,
    shuffle: bool,
) -> None:
    ipc = IPCSharedBuffer(name=ipc_name)
    shared_array = np.ndarray((qsize, batch_size), dtype=input_dtype, buffer=ipc.buf)
    local_batch_results: list[Any] = []
    lock = threading.Lock()

    try:
        while True:
            if shuffle:
                random.shuffle(data_chunk)

            def task(item: Any, end: bool = False) -> None:
                result = None if end else read_func(item)
                if end or result is not None:
                    with lock:
                        if not end and result is not None:
                            local_batch_results.append(result)
                        if local_batch_results and (end or len(local_batch_results) == batch_size):
                            idx = buffer_queue.get()
                            shared_array[idx][: len(local_batch_results)] = local_batch_results
                            ready_queue.put((idx, len(local_batch_results)))
                            local_batch_results.clear()

            with ThreadPoolExecutor(max_workers=max(1, int(no_of_worker_threads))) as executor:
                list(executor.map(task, data_chunk))

            if not infinite:
                task(None, end=True)
                break
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        ready_queue.put(None)
        ipc.close()


def _update_workers(
    action: str,
    workers: list[mp.Process],
    data: list[Any] | None = None,
    no_of_workers: int = 1,
    worker_kwargs: dict[str, Any] | None = None,
) -> None:
    if action in {"stop", "restart"}:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
                worker.join()
        workers.clear()

    if action == "restart":
        if data is None:
            raise ValueError("data must be provided for restart")
        kwargs = worker_kwargs or {}
        worker_chunks = np.array_split(data, max(1, int(no_of_workers)))
        for idx, chunk in enumerate(worker_chunks):
            process = mp.Process(
                target=_worker_loop,
                name=f"generator-worker-{idx + 1}",
                args=(chunk.tolist(),),
                kwargs=kwargs,
            )
            process.daemon = True
            process.start()
            workers.append(process)


def get_generator(
    data: list[Any],
    read_func: Callable[[Any], Any],
    input_dtype: str = "(224,224,3)f4",
    batch_size: int = 64,
    no_of_workers: int = 4,
    no_of_worker_threads: int = 8,
    qsize: int = 8,
    take: int | None = None,
    infinite: bool = False,
    shuffle: bool = False,
) -> Callable[[], Iterator[Any]]:
    """Return a callable that yields pre-batched samples via shared memory queues."""
    if not data:
        raise ValueError("data must contain at least one sample")

    dtype = np.dtype(input_dtype)
    sample_result = read_func(data[0])
    if sample_result is None:
        raise ValueError("read_func returned None for first sample")

    sample_buffer = np.zeros(1, dtype=dtype)
    try:
        sample_buffer[:] = sample_result
    except ValueError as exc:
        raise ValueError("read_func output does not match configured input_dtype") from exc

    if not infinite and len(data) // max(1, batch_size) < max(1, no_of_workers):
        no_of_workers = max(1, len(data) // max(1, batch_size))

    workers: list[mp.Process] = []

    def generator_func() -> Iterator[Any]:
        ipc = IPCSharedBuffer(size=max(1, qsize) * max(1, batch_size) * dtype.itemsize)
        shared_array = np.ndarray((qsize, batch_size), dtype=dtype, buffer=ipc.buf)
        name = ipc.name_or_path()

        buffer_queue: mp.Queue = mp.Queue()
        ready_queue: mp.Queue = mp.Queue()
        for i in range(max(1, qsize)):
            buffer_queue.put(i)

        worker_kwargs = {
            "read_func": read_func,
            "ipc_name": name,
            "buffer_queue": buffer_queue,
            "ready_queue": ready_queue,
            "batch_size": int(batch_size),
            "input_dtype": dtype,
            "qsize": int(qsize),
            "no_of_worker_threads": int(no_of_worker_threads),
            "infinite": bool(infinite),
            "shuffle": bool(shuffle),
        }

        _update_workers(
            "restart",
            workers=workers,
            data=data,
            no_of_workers=max(1, int(no_of_workers)),
            worker_kwargs=worker_kwargs,
        )

        remaining = int(take) if take is not None else (-1 if infinite else len(data))
        alive_processes = len(workers)

        try:
            while remaining != 0 and (alive_processes > 0 or ready_queue.qsize() > 0):
                try:
                    payload = ready_queue.get(timeout=30)
                except queue.Empty:
                    break

                if payload is None:
                    alive_processes -= 1
                    continue

                buffer_idx, batch_len = payload
                batched_data = shared_array[buffer_idx][:batch_len].copy()
                if dtype.fields:
                    yield tuple(np.stack(batched_data[field]) for field in dtype.fields)
                else:
                    yield batched_data
                buffer_queue.put(buffer_idx)

                if remaining > 0:
                    remaining -= 1
        finally:
            _update_workers("stop", workers=workers)
            ipc.close()
            ipc.unlink()

    return generator_func
