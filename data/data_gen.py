import os
import sys
import time
import mmap
import queue
import random
import threading
import numpy as np
from random import shuffle
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class IPCSharedBuffer:
    """
    Cross-version IPC buffer:
        - Python >= 3.8: multiprocessing.shared_memory
        - Python < 3.8: file-backed mmap (mkstemp or provided path)
    Usage:
        # Producer (create):
        buf = IPCSharedBuffer(size= 1025)
        # Consumer (attach)
        buf2 = IPCSharedBuffer(name= buf.name_or_path())
    """
    def __init__(self, size= None, name= None):
        if size is None and name is None:
            raise ValueError("provide size (to create) or name (to attach).")
        self.name = name
        self._is_shm, self.shm = False, None
        self.path, self.fd, self._mmap = None, None, None
        
        if sys.version_info >= (3, 8):
            from multiprocessing import shared_memory
            if name is None:
                self.shm = shared_memory.SharedMemory(create= True, size= size)
                self.size = self.shm.size
            else:
                self.shm = shared_memory.SharedMemory(name= name)
                try:
                    self.size = self.shm.size
                except Exception:
                    if self.size is None: raise ValueError("size is unknown for this SharedMemory; pass size explicitly")
            self._is_shm = True
            self._buffer = self.shm.buf
        else:
            # Fallback: file-backed mmap on tmpfs (/dev/shm if present)
            tmpdir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
            if name is None:
                fd, path = os.mkstemp(dir= tmpdir)
                os.ftruncate(fd, size)
            else:
                path = name
                if os.path.exists(path):
                    fd = os.open(path, O_RDWR)
            self.fd, self.path = fd, path
            self.size = os.stat(fd).st_size
            self._mmap = mmap.mmap(self.fd, self.size)
            self._buffer = memoryview(self._mmap)
    
    @property
    def buf(self):
        return self._buffer
    
    def name_or_path(self):
        if self._is_shm:
            return self.shm.name
        return self.path
    
    def close(self):
        if self._is_shm:
            try: self.shm.close()
            except Exception: pass
        else:
            if self._buffer is not None:
                try: self._buffer.release()
                except Exception: pass
            if self._mmap is not None:
                try: self._mmap.flush(), self._mmap.close()
                except Exception: pass
            if self.fd is not None:
                try: os.close(self.fd)
                except Exception: pass
    
    def unlink(self):
        if self._is_shm:
            try: self.shm.unlink()
            except FileNotFoundError: pass
        else:
            if self.path:
                try: os.unlink(self.path)
                except FileNotFoundError: pass
                
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.close()

def __worker_func(data_chunk, 
                  read_func, 
                  ipc_name, 
                  buffer_queue, ready_queue,
                  batch_size, input_dtypes,
                  qsize, no_of_worker_threads,
                  infinite, shuffle):
    try:
        
        ipc = IPCSharedBuffer(name= ipc_name)
        shared_array = np.ndarray((qsize, batch_size), dtype= input_dtypes, buffer= ipc.buf)

        local_batch_results = []
        lock = threading.Lock()
        while True:
            if shuffle:
                random.shuffle(data_chunk)

            def task(x, end= False):
                if end is False:
                    result = read_func(x)
                if end is True or result is not None:
                    with lock:
                        if end is False and result is not None:
                            local_batch_results.append(result)
                        if len(local_batch_results) > 0 and (end is True or len(local_batch_results) == batch_size):
                            idx = buffer_queue.get()
                            shared_array[idx][:len(local_batch_results)] = local_batch_results
                            ready_queue.put((idx, len(local_batch_results)))
                            local_batch_results.clear()

            with ThreadPoolExecutor(max_workers= no_of_worker_threads) as executor:
                _ = executor.map(task, data_chunk)

            if not infinite:
                time.sleep(5)
                task(None, end= True)
                break
    except:
        import traceback
        traceback.print_exc()
            
    ready_queue.put(None)

def __modify_worker_processess_status(
        action, workers= None, data= None, no_of_workers=1, worker_func_kwargs= None
    ):
    if action in "stop|restart":
        for worker in workers:
            if isinstance(worker, mp.Process) and worker.is_alive():
                worker.kill()
                worker.join()
    workers.clear()
    if action in "restart":
        worker_func_kwargs = worker_func_kwargs or {}
        worker_chunks = np.array_split(data, no_of_workers)
        for j, _worker_chunk in enumerate(worker_chunks):
            workers.append(
                mp.Process(
                    target=__worker_func,
                    name=f"dataloader worker #{j+1}",
                    args=(_worker_chunk, ),
                    kwargs=worker_func_kwargs,
                )
            )
            workers[j].daemon = True
            workers[j].start()



def getGenerator(data,
                 read_func,
                 input_dtypes= "(224, 224, 3)f4",
                 batch_size=512,
                 no_of_workers=10,
                 no_of_worker_threads=64,
                 qsize=10,
                 take=None,
                 infinite=False,
                 shuffle=False):
    
    input_dtypes= np.dtype(input_dtypes)
    sample_result = read_func(data[0])
    if sample_result is None:
        raise ValueError("read func returned None for sample data[0], check the func")
    sample_array_buffer = np.zeros(1, dtype= input_dtypes)
    try:
        if len(input_dtypes) != (len(sample_result) if isinstance(sample_result, tuple) else 0):
            raise ValueError("len of read_func returned value doesn't match with len of input_dtypes")
        sample_array_buffer[:] = sample_result
    except ValueError:
        print("read_func returned value dtype/shape doesn't match with provided input_dtypes inner dimentions")
        raise
    
    if (not infinite) and (segment_size := len(data) // batch_size) < no_of_workers:
        no_of_workers = np.cumsum([batch_size] * (segment_size + 1))
    worker_processes = []
    
    def generator_func():

        ipc = IPCSharedBuffer(qsize * batch_size * input_dtypes.itemsize)
        shared_array = np.ndarray((qsize, batch_size), dtype= input_dtypes, buffer= ipc.buf)
        name = ipc.name_or_path()
    
        buffer_queue = mp.Queue()
        ready_queue = mp.Queue()
        for i in range(qsize):
            buffer_queue.put(i)

        worker_func_kwargs = {
            "read_func": read_func,
            "ipc_name": name,
            "buffer_queue": buffer_queue, "ready_queue": ready_queue,
            "batch_size": batch_size, "input_dtypes": input_dtypes,
            "qsize": qsize,
            "no_of_worker_threads": no_of_worker_threads,
            "infinite": infinite,
            "shuffle": shuffle
        }
        
        __modify_worker_processess_status("restart", worker_processes, data, no_of_workers, worker_func_kwargs)
        count = take if take is not None else -1 if infinite else len(data)
        alive_process = len(worker_processes)
        try:
            while count and (alive_process > 0 or ready_queue.qsize() > 0):
                ready_q_data = ready_queue.get(timeout=30)
                if ready_q_data is None:
                    alive_process -= 1
                    continue
                idx, len_batch = ready_q_data
                batched_data = shared_array[idx][:len_batch].copy()
                yield tuple(np.stack(batched_data[f]) for f in batched_data.dtype.fields) if len(batched_data.dtype) else batched_data
                buffer_queue.put(idx)
                count -= 1
        except queue.Empty as e:
            return

        finally:
            __modify_worker_processess_status("stop", worker_processes)
            ipc.close()
            ipc.unlink()
            
    return generator_func
