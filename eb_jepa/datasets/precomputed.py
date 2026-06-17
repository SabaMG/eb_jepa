"""GPU-resident pre-computed dataset with double-buffered async generation.

Continuous pipeline (step-level swap variant — Pack 4):
  - VRAM permanently holds two small chunks: `current` (training reads from it)
    and `next` (already transferred, waiting).
  - Workers on 116 CPU cores are always generating the chunk-after-next.
  - Every `batches_per_chunk` training steps: wait for the in-flight transfer to
    land, promote `next → current`, kick off a new async transfer, and start the
    next generation immediately.

With chunk_size=3840 (10 batches × 384) and ~116 workers:
  - generation time ≈ 1.71s  (3840 samples / ~2250 samples·s⁻¹)
  - training time  ≈ 1.88s  (10 batches, same GPU throughput as Pack 3)
  → generation finishes before the GPU needs the next chunk.

Memory design (same as Pack 3):
  - Workers cast tensors to target dtype immediately (halves worker RAM).
  - collect() pre-allocates the output buffer and fills one part at a time.
  - warm_up() explicitly deletes each CPU copy after the GPU transfer completes.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import torch

from eb_jepa.datasets.two_rooms.normalizer import Normalizer
from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig
from eb_jepa.logging import get_logger

logger = get_logger(__name__)

_WORKER_DATASET = None

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_dataset(env_name):
    """Lazy lookup of (DatasetClass, ConfigClass) for a worker process.

    Imports are done inside the function so each env module is only loaded
    when actually used (and each loads its own dependencies).
    """
    if env_name == "two_rooms":
        from eb_jepa.datasets.two_rooms.wall_dataset import (
            WallDataset as DatasetClass,
        )
        from eb_jepa.datasets.two_rooms.wall_dataset import (
            WallDatasetConfig as ConfigClass,
        )
    elif env_name == "maze":
        from eb_jepa.datasets.maze.maze_dataset import (
            MazeDataset as DatasetClass,
        )
        from eb_jepa.datasets.maze.maze_dataset import (
            MazeDatasetConfig as ConfigClass,
        )
    else:
        raise ValueError(
            f"Unknown env_name={env_name!r}; expected 'two_rooms' or 'maze'"
        )
    return DatasetClass, ConfigClass


def _worker_init(env_name, env_config_dict):
    global _WORKER_DATASET
    DatasetClass, ConfigClass = _resolve_dataset(env_name)
    cfg_for_worker = dict(env_config_dict)
    cfg_for_worker["device"] = "cpu"
    config = update_config_from_yaml(ConfigClass, cfg_for_worker)
    _WORKER_DATASET = DatasetClass(config=config)


def _generate_part(seed, n_samples, dtype_name=None):
    import random

    import numpy as np

    global _WORKER_DATASET
    assert _WORKER_DATASET is not None, "Worker not initialized"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    dtype = _DTYPE_MAP.get(dtype_name) if dtype_name else None

    states, actions, locations, wall_x, door_y = [], [], [], [], []
    for _ in range(n_samples):
        s = _WORKER_DATASET.generate_multistep_sample()
        st = s.states.squeeze(0)
        ac = s.actions.squeeze(0)
        lo = s.locations.squeeze(0)
        if dtype is not None:
            if st.dtype != dtype:
                st = st.to(dtype)
            if ac.dtype != dtype:
                ac = ac.to(dtype)
            if lo.dtype != dtype:
                lo = lo.to(dtype)
        states.append(st)
        actions.append(ac)
        locations.append(lo)
        wall_x.append(s.wall_x)
        door_y.append(s.door_y)

    return {
        "states": torch.stack(states, dim=0),
        "actions": torch.stack(actions, dim=0),
        "locations": torch.stack(locations, dim=0),
        "wall_x": torch.stack(wall_x, dim=0),
        "door_y": torch.stack(door_y, dim=0),
    }


class AsyncChunkGenerator:
    """Persistent pool of CPU workers that produce chunks of samples in parallel."""

    def __init__(self, env_name, env_config_dict, num_workers, dtype=None):
        self.num_workers = num_workers
        _rev = {v: k for k, v in _DTYPE_MAP.items()}
        self._dtype_name = _rev.get(dtype)
        ctx = get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(env_name, env_config_dict),
        )

    def submit(self, chunk_id, chunk_size):
        per = chunk_size // self.num_workers
        rem = chunk_size - per * self.num_workers
        seed_base = (chunk_id + 1) * 1_000_003
        futures = []
        for i in range(self.num_workers):
            n = per + (1 if i < rem else 0)
            if n > 0:
                futures.append(
                    self.executor.submit(_generate_part, seed_base + i, n, self._dtype_name)
                )
        return futures

    @staticmethod
    def collect(futures, total_size=None):
        """Streaming pre-allocated collection — peak RAM = output + one worker part."""
        first = futures[0].result()
        n_first = first["states"].shape[0]
        n_total = total_size if total_size is not None else n_first * len(futures)

        output = {
            k: torch.empty(n_total, *v.shape[1:], dtype=v.dtype)
            for k, v in first.items()
        }
        for k in output:
            output[k][:n_first] = first[k]
        del first

        offset = n_first
        for f in futures[1:]:
            part = f.result()
            n = part["states"].shape[0]
            for k in output:
                output[k][offset : offset + n] = part[k]
            del part
            offset += n

        return output

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


class _DatasetView:
    """Minimal stand-in for `loader.dataset` exposing `.normalizer` and `__len__`."""

    def __init__(self, normalizer, size):
        self.normalizer = normalizer
        self._size = size

    def __len__(self):
        return self._size


class PipelineManager:
    """Owns the double-buffered VRAM chunks and orchestrates async refill.

    The chunk *source* is injected via ``generator``: any object exposing the
    ``submit(chunk_id, chunk_size)`` / ``collect(handle, total_size)`` /
    ``shutdown()`` contract works. Defaults to ``AsyncChunkGenerator`` (online
    CPU generation); pass e.g. an ``OfflineChunkReader`` to stream a
    pre-generated dataset from disk through the exact same double-buffered
    VRAM-staging machinery.
    """

    def __init__(self, env_config_dict, chunk_size, device, dtype, num_workers,
                 generator=None, env_name="two_rooms"):
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype
        self.generator = (
            generator
            if generator is not None
            else AsyncChunkGenerator(env_name, env_config_dict, num_workers, dtype=dtype)
        )
        self.transfer_stream = torch.cuda.Stream(device=device)
        self._transfer_executor = ThreadPoolExecutor(max_workers=1)

        self.current = None
        self.next = None

        self._pending_gen_futures = None
        self._pending_transfer_future = None
        self._next_chunk_id = 0

    def _to_device_blocking(self, cpu_chunk):
        return {k: v.to(self.device, non_blocking=False) for k, v in cpu_chunk.items()}

    def _transfer_async_thread(self, cpu_chunk):
        with torch.cuda.stream(self.transfer_stream):
            pinned = {k: v.pin_memory() for k, v in cpu_chunk.items()}
            gpu_chunk = {k: v.to(self.device, non_blocking=True) for k, v in pinned.items()}
        self.transfer_stream.synchronize()
        del cpu_chunk
        return gpu_chunk

    def warm_up(self):
        """Generate the first two chunks and pre-fill `current` and `next`.

        Sequential with explicit CPU memory release between transfers so both
        ~536 MB buffers never overlap. Kicks off chunk 2 generation before returning.
        """
        f0 = self.generator.submit(self._next_chunk_id, self.chunk_size)
        chunk0 = self.generator.collect(f0, total_size=self.chunk_size)
        self.current = self._to_device_blocking(chunk0)
        del chunk0
        self._next_chunk_id += 1

        f1 = self.generator.submit(self._next_chunk_id, self.chunk_size)
        chunk1 = self.generator.collect(f1, total_size=self.chunk_size)
        self.next = self._to_device_blocking(chunk1)
        del chunk1
        self._next_chunk_id += 1

        self._pending_gen_futures = self.generator.submit(
            self._next_chunk_id, self.chunk_size
        )

    def swap(self):
        """Step-boundary swap: promote `next → current` and refill `next` async."""
        if self._pending_transfer_future is not None:
            self.next = self._pending_transfer_future.result()
            self._pending_transfer_future = None

        self.current = self.next
        self.next = None

        cpu_chunk = self.generator.collect(
            self._pending_gen_futures, total_size=self.chunk_size
        )
        self._pending_gen_futures = None

        self._pending_transfer_future = self._transfer_executor.submit(
            self._transfer_async_thread, cpu_chunk
        )

        self._next_chunk_id += 1
        self._pending_gen_futures = self.generator.submit(
            self._next_chunk_id, self.chunk_size
        )

    def shutdown(self):
        self._transfer_executor.shutdown(wait=False, cancel_futures=True)
        self.generator.shutdown()


class PipelineLoader:
    """DataLoader-shaped iterator that performs step-level double-buffer swaps.

    Each call to `__iter__` yields `epoch_size // batch_size` batches total.
    Internally it consumes `chunk_size // batch_size` batches from VRAM, then
    calls `manager.swap()` to promote the pre-fetched next chunk, repeating
    until the epoch is complete.

    The very first chunk of the very first epoch skips the swap (warm_up already
    filled `current`). Every subsequent chunk — including the first chunk of
    each subsequent epoch — triggers a swap so training always sees fresh data.
    """

    def __init__(self, manager, batch_size, epoch_size, drop_last=True, normalizer=None,
                 shuffle=True):
        self.manager = manager
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.drop_last = drop_last
        # shuffle=False traverses each chunk in stored order — used for the
        # offline-stream source, where the dataset is already i.i.d. on disk so
        # no per-chunk reshuffle is needed (and order is deterministic).
        self.shuffle = shuffle
        self.dataset = _DatasetView(
            normalizer=normalizer if normalizer is not None else Normalizer(),
            size=epoch_size,
        )
        self._first_iter = True

    def __len__(self):
        return self.epoch_size // self.batch_size

    def __iter__(self):
        batches_per_chunk = self.manager.chunk_size // self.batch_size
        n_chunks = len(self) // batches_per_chunk

        for chunk_idx in range(n_chunks):
            if self._first_iter and chunk_idx == 0:
                self._first_iter = False  # use warm_up'd current chunk as-is
            else:
                self.manager.swap()

            current = self.manager.current
            n = current["states"].size(0)
            if self.shuffle:
                indices = torch.randperm(n, device=current["states"].device)
            else:
                indices = torch.arange(n, device=current["states"].device)

            for i in range(batches_per_chunk):
                sel = indices[i * self.batch_size : (i + 1) * self.batch_size]
                yield (
                    current["states"].index_select(0, sel),
                    current["actions"].index_select(0, sel),
                    current["locations"].index_select(0, sel),
                    current["wall_x"].index_select(0, sel),
                    current["door_y"].index_select(0, sel),
                )


def init_precomputed_data(
    env_config_dict,
    chunk_size,
    epoch_size,
    batch_size,
    device,
    dtype,
    num_workers,
    drop_last=True,
    env_name="two_rooms",
    normalizer=None,
):
    """Build the pipeline manager and a PipelineLoader. Caller must invoke
    `manager.warm_up()` once before iterating the loader.

    Args:
        chunk_size: samples per swap (small, e.g. 3840 = 10 batches × 384).
        epoch_size: total samples per epoch (e.g. 100000); must be divisible by
            chunk_size * batch_size / batch_size = chunk_size (or rather,
            epoch_size // batch_size must be divisible by chunk_size // batch_size).

    Returns:
        loader: PipelineLoader yielding (x, a, loc, wall_x, door_y) on GPU.
        manager: PipelineManager (caller owns lifecycle, must call shutdown()).
    """
    manager = PipelineManager(
        env_name=env_name,
        env_config_dict=env_config_dict,
        chunk_size=chunk_size,
        device=device,
        dtype=dtype,
        num_workers=num_workers,
    )
    loader = PipelineLoader(
        manager=manager,
        batch_size=batch_size,
        epoch_size=epoch_size,
        drop_last=drop_last,
        normalizer=normalizer if normalizer is not None else Normalizer(),
    )
    return loader, manager
