"""Offline (pre-computed) two-rooms dataset.

Pack 8 splits data generation from training. Instead of regenerating fresh
samples online inside every ``__getitem__`` (the base ``WallDataset`` behaviour),
a *preprocessing* step generates a fixed dataset of ``num_samples`` trajectories
once and writes them to disk as memory-mapped arrays. Training then iterates the
static dataset with the standard ``DataLoader`` pipeline — same per-step batch
loading, steps grouped into epochs — exactly like the base loop, just reading
from disk instead of synthesising on the fly.

Two generation backends, selectable at preprocessing time:

  - ``cpu``: a pool of worker processes, each running the *verbatim* base
    ``WallDataset.generate_multistep_sample`` (one sample at a time, identical to
    the online path), filling disjoint index ranges of the memmaps.
  - ``gpu``: the vectorised ``GPUWallGenerator`` (a port of the same generator),
    producing whole batches directly on the GPU.

Storage format (one directory per dataset):

  - ``states.dat``    float16, shape (N, 2, sample_length, H, W)   — normalised
  - ``actions.dat``   float32, shape (N, 2, sample_length)         — raw
  - ``locations.dat`` float32, shape (N, 2, sample_length)         — normalised
  - ``wall_x.dat``    float32, shape (N,)
  - ``door_y.dat``    float32, shape (N,)
  - ``meta.json``     shapes / dtypes / generation parameters

States are stored *already normalised and in the final (C, T, H, W) layout*, i.e.
exactly what the base ``WallDataset.__getitem__`` returns. The loader therefore
applies no transform — it just maps the bytes and yields tensors — which keeps it
bit-faithful to the online dataset's output and free of any normalisation/layout
ambiguity. Cost is 2x the raw-uint8 footprint (~344 GB at N=1.2M), which is fine
on the 10 TB work filesystem.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch

from eb_jepa.datasets.two_rooms.normalizer import Normalizer
from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

STATES_FILE = "states.dat"
ACTIONS_FILE = "actions.dat"
LOCATIONS_FILE = "locations.dat"
WALL_X_FILE = "wall_x.dat"
DOOR_Y_FILE = "door_y.dat"
META_FILE = "meta.json"

STATES_DTYPE = np.float16
ARR_DTYPE = np.float32


# --------------------------------------------------------------------------- #
# Dataset (training reads this)
# --------------------------------------------------------------------------- #
class OfflineWallDataset(torch.utils.data.Dataset):
    """Memory-mapped, pre-computed two-rooms dataset.

    Returns the same 5-tuple as the base ``WallDataset`` so the training loop
    consumes it unchanged: ``(states, actions, locations, wall_x, door_y)``.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, META_FILE)) as f:
            self.meta = json.load(f)

        self.num_samples = int(self.meta["num_samples"])
        self.states_shape = tuple(self.meta["states_shape"])
        self.actions_shape = tuple(self.meta["actions_shape"])
        self.locations_shape = tuple(self.meta["locations_shape"])

        # Lazily opened per-worker (memmap handles are not fork-safe to share).
        self._states = None
        self._actions = None
        self._locations = None
        self._wall_x = None
        self._door_y = None

        # main.py reads ``loader.dataset.normalizer`` for the probe loss.
        self.normalizer = Normalizer()

    def _ensure_open(self):
        if self._states is not None:
            return
        d = self.data_dir
        n = self.num_samples
        self._states = np.memmap(
            os.path.join(d, STATES_FILE), dtype=STATES_DTYPE, mode="r",
            shape=(n, *self.states_shape),
        )
        self._actions = np.memmap(
            os.path.join(d, ACTIONS_FILE), dtype=ARR_DTYPE, mode="r",
            shape=(n, *self.actions_shape),
        )
        self._locations = np.memmap(
            os.path.join(d, LOCATIONS_FILE), dtype=ARR_DTYPE, mode="r",
            shape=(n, *self.locations_shape),
        )
        self._wall_x = np.memmap(
            os.path.join(d, WALL_X_FILE), dtype=ARR_DTYPE, mode="r", shape=(n,)
        )
        self._door_y = np.memmap(
            os.path.join(d, DOOR_Y_FILE), dtype=ARR_DTYPE, mode="r", shape=(n,)
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        self._ensure_open()
        # ``.copy()`` lifts the slice out of the memmap into owned memory so the
        # returned tensors don't pin the mapping.
        states = torch.from_numpy(self._states[i].copy()).float()
        actions = torch.from_numpy(self._actions[i].copy())
        locations = torch.from_numpy(self._locations[i].copy())
        wall_x = torch.tensor(float(self._wall_x[i]))
        door_y = torch.tensor(float(self._door_y[i]))
        return states, actions, locations, wall_x, door_y


# --------------------------------------------------------------------------- #
# Streaming source (reads pre-generated data through the stream pipeline)
# --------------------------------------------------------------------------- #
class OfflineChunkReader:
    """Chunk source for ``PipelineManager`` that reads a pre-generated dataset.

    Implements the same ``submit`` / ``collect`` / ``shutdown`` contract as
    ``AsyncChunkGenerator`` (the online CPU generator), so it plugs into the
    existing double-buffered, VRAM-staging ``PipelineManager`` unchanged — the
    only difference is the source of bytes: large **sequential** reads from the
    memmap on disk instead of per-sample CPU generation.

    This fixes the naive offline ``DataLoader`` path, whose ``shuffle=True``
    per-sample ``__getitem__`` turns one epoch into ~1.2M random small reads
    scattered across a multi-hundred-GB memmap on Lustre (I/O-bound, ~25 min/
    epoch). Sequential chunk reads + on-GPU iteration are dramatically faster.

    Read order is controlled by ``shuffle``:
      - ``False`` (default): samples are traversed in stored order. ``chunk_id``
        increments forever across epochs, so the read offset wraps modulo
        ``num_samples``; a chunk spanning the end is stitched from tail + head.
        Deterministic, identical every epoch.
      - ``True``: *block shuffle* — the chunk order is randomly permuted once
        per epoch (each chunk read exactly once, in random order), and the
        ``PipelineLoader`` additionally shuffles within each chunk. This keeps
        the large sequential per-chunk reads (fast on Lustre) while giving fresh
        per-epoch randomisation. Since the dataset is already i.i.d. on disk,
        block shuffle is statistically ≈ a full shuffle, without reintroducing
        slow random small reads.
    """

    def __init__(self, data_dir, dtype=None, shuffle=False, read_workers=1):
        self.dataset = OfflineWallDataset(data_dir)
        self.dataset._ensure_open()
        self.num_samples = self.dataset.num_samples
        self.dtype = dtype
        self.shuffle = shuffle
        self._perm = None          # permuted chunk indices for the current epoch
        self._num_chunks = None
        self._rng = np.random.default_rng()
        # Background submit thread: overlaps the disk read of the next chunk with
        # training on the current one (mirrors AsyncChunkGenerator's async submit).
        self._executor = ThreadPoolExecutor(max_workers=1)
        # Intra-chunk parallelism: a chunk read of `n` samples is split into
        # `read_workers` disjoint sub-ranges read concurrently. numpy releases
        # the GIL during the bulk memmap->array memcpy, so these threads issue
        # multiple outstanding I/O requests against Lustre at once — the fix for
        # the ~0.1 GB/s single-threaded read on the stripe_count=1 dataset.
        self.read_workers = max(1, int(read_workers))
        self._read_pool = (
            ThreadPoolExecutor(max_workers=self.read_workers)
            if self.read_workers > 1
            else None
        )

    def _read_into(self, mm, out, lo, hi, start):
        """Read absolute samples [start+lo, start+hi) of ``mm`` into ``out[lo:hi]``.

        Reads straight from the memmap into the pre-allocated owned buffer
        ``out`` (no intermediate copy), wrapping around the end of the dataset.
        """
        a = (start + lo) % self.num_samples
        length = hi - lo
        end = a + length
        if end <= self.num_samples:
            out[lo:hi] = mm[a:end]
        else:
            first = self.num_samples - a
            out[lo : lo + first] = mm[a:]
            out[lo + first : hi] = mm[: length - first]

    def _read_array(self, mm, start, n):
        """Owned copy of ``n`` samples from ``mm`` starting at ``start`` (wrapping).

        With ``read_workers > 1`` the read is fanned out over disjoint sub-ranges
        so multiple I/O requests hit the OST concurrently.
        """
        out = np.empty((n, *mm.shape[1:]), dtype=mm.dtype)
        if self._read_pool is None or n < self.read_workers:
            self._read_into(mm, out, 0, n, start)
            return out
        nb = self.read_workers
        bounds = [(i * n // nb, (i + 1) * n // nb) for i in range(nb)]
        futs = [
            self._read_pool.submit(self._read_into, mm, out, lo, hi, start)
            for lo, hi in bounds
            if hi > lo
        ]
        for f in futs:
            f.result()
        return out

    def _read_chunk(self, start, n):
        ds = self.dataset
        # The big three arrays (states dominates) are read with intra-chunk
        # parallelism; wall_x/door_y are tiny so read serially.
        chunk = {
            "states": torch.from_numpy(self._read_array(ds._states, start, n)),
            "actions": torch.from_numpy(self._read_array(ds._actions, start, n)),
            "locations": torch.from_numpy(self._read_array(ds._locations, start, n)),
            "wall_x": torch.from_numpy(self._read_array(ds._wall_x, start, n)),
            "door_y": torch.from_numpy(self._read_array(ds._door_y, start, n)),
        }
        # Match the stream pipeline's dtype convention (states/actions/locations
        # cast to the training dtype; wall_x/door_y stay float32).
        if self.dtype is not None:
            for k in ("states", "actions", "locations"):
                chunk[k] = chunk[k].to(self.dtype)
        return chunk

    def _chunk_start(self, chunk_id, chunk_size):
        if not self.shuffle:
            # Sequential, wrapping across epochs.
            return (chunk_id * chunk_size) % self.num_samples
        # Block shuffle: permute whole-chunk indices, reshuffling each epoch.
        if self._num_chunks is None:
            self._num_chunks = max(1, self.num_samples // chunk_size)
        epoch_local = chunk_id % self._num_chunks
        if epoch_local == 0:
            self._perm = self._rng.permutation(self._num_chunks)
        return int(self._perm[epoch_local]) * chunk_size

    def submit(self, chunk_id, chunk_size):
        start = self._chunk_start(chunk_id, chunk_size)
        return self._executor.submit(self._read_chunk, start, chunk_size)

    @staticmethod
    def collect(handle, total_size=None):
        # The reader returns a full, correctly-sized chunk; total_size is part of
        # the contract (AsyncChunkGenerator needs it) but unused here.
        return handle.result()

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._read_pool is not None:
            self._read_pool.shutdown(wait=False, cancel_futures=True)


class DeepPrefetchManager:
    """Continuous-prefetch VRAM manager for the offline-stream path.

    Drop-in replacement for ``PipelineManager`` (same ``chunk_size`` / ``current``
    / ``warm_up`` / ``swap`` / ``shutdown`` surface used by ``PipelineLoader``),
    but instead of a single double-buffer it keeps a *queue* of
    ``prefetch_depth`` chunks continuously being read from disk and staged into
    VRAM, never blocking training as long as the disk keeps up. Each consumed GPU
    chunk is dropped (``del``) the instant its batches are exhausted, so VRAM
    holds at most ``prefetch_depth + 1`` chunks at a time.

    Pipeline invariant (after warm_up and after every swap):
      - ``read_handles``    : ``prefetch_depth`` disk reads submitted, not yet collected
      - ``transfer_futures``: ``prefetch_depth`` H2D transfers in flight / ready
      - ``current``         : the one GPU chunk training is reading from
    """

    def __init__(self, reader, chunk_size, device, dtype, prefetch_depth=3):
        self.generator = reader
        self.chunk_size = chunk_size
        self.device = device
        self.dtype = dtype
        self.prefetch_depth = max(1, int(prefetch_depth))
        # One CUDA stream per transfer thread: a single shared stream would force
        # the concurrent threads to serialise on its .synchronize().
        self._streams = [
            torch.cuda.Stream(device=device) for _ in range(self.prefetch_depth)
        ]
        self._transfer_executor = ThreadPoolExecutor(max_workers=self.prefetch_depth)
        self._stream_rr = 0
        self.current = None
        self._read_handles = deque()
        self._transfer_futures = deque()
        self._next_chunk_id = 0

    def _submit_read(self):
        self._read_handles.append(
            self.generator.submit(self._next_chunk_id, self.chunk_size)
        )
        self._next_chunk_id += 1

    def _transfer_async_thread(self, cpu_chunk, stream):
        with torch.cuda.stream(stream):
            pinned = {k: v.pin_memory() for k, v in cpu_chunk.items()}
            gpu_chunk = {
                k: v.to(self.device, non_blocking=True) for k, v in pinned.items()
            }
        stream.synchronize()
        del cpu_chunk
        return gpu_chunk

    def _start_transfer(self):
        handle = self._read_handles.popleft()
        cpu_chunk = self.generator.collect(handle, total_size=self.chunk_size)
        stream = self._streams[self._stream_rr % self.prefetch_depth]
        self._stream_rr += 1
        self._transfer_futures.append(
            self._transfer_executor.submit(
                self._transfer_async_thread, cpu_chunk, stream
            )
        )

    def warm_up(self):
        # Prime the whole pipeline: enough reads queued to feed `prefetch_depth`
        # in-flight transfers plus the one we promote to `current`.
        for _ in range(2 * self.prefetch_depth):
            self._submit_read()
        for _ in range(self.prefetch_depth):
            self._start_transfer()
        self.current = self._transfer_futures.popleft().result()
        # Restore the invariant (depth transfers in flight, depth reads queued).
        self._start_transfer()
        self._submit_read()

    def swap(self):
        # Free the exhausted GPU chunk immediately, then promote the next ready one.
        self.current = None
        self.current = self._transfer_futures.popleft().result()
        self._start_transfer()
        self._submit_read()

    def shutdown(self):
        self._transfer_executor.shutdown(wait=False, cancel_futures=True)
        self.generator.shutdown()


def init_offline_stream_data(
    data_dir,
    chunk_size,
    batch_size,
    device,
    dtype,
    epoch_size=None,
    drop_last=True,
    shuffle=False,
    read_workers=1,
    prefetch_depth=1,
):
    """Stream a pre-generated dataset through the VRAM-staging stream pipeline.

    Reuses ``PipelineLoader`` (on-GPU batch slicing) verbatim; the chunk source
    is ``OfflineChunkReader`` (with optional intra-chunk parallel reads via
    ``read_workers``). One epoch = one full pass over ``epoch_size`` samples.
    ``shuffle`` selects sequential (False) vs block-shuffle (True) traversal.

    The VRAM manager depends on ``prefetch_depth``:
      - ``1``: the classic double-buffer ``PipelineManager`` (current + next).
      - ``>1``: ``DeepPrefetchManager`` keeps ``prefetch_depth`` chunks reading
        and staging continuously, freeing each GPU chunk as it is consumed.

    Caller must call ``manager.warm_up()`` before iterating and
    ``manager.shutdown()`` at the end (same lifecycle as the online stream).
    """
    from eb_jepa.datasets.precomputed import PipelineLoader, PipelineManager

    reader = OfflineChunkReader(
        data_dir, dtype=dtype, shuffle=shuffle, read_workers=read_workers
    )
    if epoch_size is None:
        epoch_size = reader.num_samples
    if int(prefetch_depth) > 1:
        manager = DeepPrefetchManager(
            reader=reader,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
            prefetch_depth=int(prefetch_depth),
        )
    else:
        manager = PipelineManager(
            env_config_dict=None,
            chunk_size=chunk_size,
            device=device,
            dtype=dtype,
            num_workers=0,
            generator=reader,
        )
    loader = PipelineLoader(
        manager=manager,
        batch_size=batch_size,
        epoch_size=epoch_size,
        drop_last=drop_last,
        normalizer=Normalizer(),
        shuffle=shuffle,
    )
    return loader, manager


# --------------------------------------------------------------------------- #
# Preprocessing (generation -> disk)
# --------------------------------------------------------------------------- #
def _open_memmaps(data_dir, n, states_shape, actions_shape, locations_shape, mode):
    states = np.memmap(
        os.path.join(data_dir, STATES_FILE), dtype=STATES_DTYPE, mode=mode,
        shape=(n, *states_shape),
    )
    actions = np.memmap(
        os.path.join(data_dir, ACTIONS_FILE), dtype=ARR_DTYPE, mode=mode,
        shape=(n, *actions_shape),
    )
    locations = np.memmap(
        os.path.join(data_dir, LOCATIONS_FILE), dtype=ARR_DTYPE, mode=mode,
        shape=(n, *locations_shape),
    )
    wall_x = np.memmap(
        os.path.join(data_dir, WALL_X_FILE), dtype=ARR_DTYPE, mode=mode, shape=(n,)
    )
    door_y = np.memmap(
        os.path.join(data_dir, DOOR_Y_FILE), dtype=ARR_DTYPE, mode=mode, shape=(n,)
    )
    return states, actions, locations, wall_x, door_y


def _shapes_from_config(cfg):
    sl = cfg.sample_length
    return (2, sl, cfg.img_size, cfg.img_size), (2, sl), (2, sl)


def _cpu_worker(args):
    """Generate a contiguous index range with the base per-sample CPU generator."""
    data_dir, start, count, seed, merged_cfg, shapes, n = args
    states_shape, actions_shape, locations_shape = shapes

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)
    dset = WallDataset(config=config)

    states, actions, locations, wall_x, door_y = _open_memmaps(
        data_dir, n, states_shape, actions_shape, locations_shape, mode="r+"
    )

    for j in range(count):
        sample = dset.generate_multistep_sample()
        idx = start + j
        states[idx] = sample.states[0].to(torch.float16).numpy()
        actions[idx] = sample.actions[0].to(torch.float32).numpy()
        locations[idx] = sample.locations[0].to(torch.float32).numpy()
        wall_x[idx] = float(sample.wall_x[0])
        door_y[idx] = float(sample.door_y[0])

    states.flush()
    actions.flush()
    locations.flush()
    wall_x.flush()
    door_y.flush()
    return count


def _generate_cpu(data_dir, n, merged_cfg, shapes, num_workers, seed, chunk):
    """Fan out generation over ``num_workers`` processes, disjoint index ranges."""
    tasks = []
    start = 0
    task_id = 0
    while start < n:
        count = min(chunk, n - start)
        # Distinct seed per task so the ranges are independent (the base
        # generator draws a fresh wall layout + trajectory per sample).
        tasks.append(
            (data_dir, start, count, seed + 1 + task_id, merged_cfg, shapes, n)
        )
        start += count
        task_id += 1

    done = 0
    t0 = time.time()
    # "spawn" rather than the default "fork": fork after CUDA has been
    # initialised in the parent (e.g. the smoke test runs the GPU backend first)
    # deadlocks the children. spawn starts clean workers, robust either way.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futures = [ex.submit(_cpu_worker, t) for t in tasks]
        for fut in as_completed(futures):
            done += fut.result()
            rate = done / max(time.time() - t0, 1e-6)
            print(
                f"[cpu] {done}/{n} samples ({rate:.0f}/s)", flush=True
            )


def _generate_gpu(data_dir, n, merged_cfg, shapes, gen_batch_size, seed):
    from eb_jepa.datasets.two_rooms.gpu_generator import GPUWallGenerator

    states_shape, actions_shape, locations_shape = shapes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)
    gen = GPUWallGenerator(config, device=device)

    states, actions, locations, wall_x, door_y = _open_memmaps(
        data_dir, n, states_shape, actions_shape, locations_shape, mode="r+"
    )

    bs = gen_batch_size or 2048
    done = 0
    t0 = time.time()
    while done < n:
        b = min(bs, n - done)
        batch = gen.generate_batch(b)
        sl = slice(done, done + b)
        states[sl] = batch["states"].to(torch.float16).cpu().numpy()
        actions[sl] = batch["actions"].to(torch.float32).cpu().numpy()
        locations[sl] = batch["locations"].to(torch.float32).cpu().numpy()
        wall_x[sl] = batch["wall_x"].to(torch.float32).cpu().numpy()
        door_y[sl] = batch["door_y"].to(torch.float32).cpu().numpy()
        done += b
        rate = done / max(time.time() - t0, 1e-6)
        print(f"[gpu] {done}/{n} samples ({rate:.0f}/s)", flush=True)

    states.flush()
    actions.flush()
    locations.flush()
    wall_x.flush()
    door_y.flush()


def generate_and_save(
    data_dir,
    num_samples,
    backend="gpu",
    merged_cfg=None,
    num_workers=16,
    gen_batch_size=2048,
    seed=0,
):
    """Generate ``num_samples`` two-rooms samples and write them to ``data_dir``.

    ``merged_cfg`` is the two-rooms data config dict (defaults loaded by the
    caller from ``data_config.yaml``); ``normalize`` is forced on so the stored
    states match the base online dataset's output.
    """
    from eb_jepa.datasets.utils import load_env_data_config

    if merged_cfg is None:
        merged_cfg = load_env_data_config("two_rooms")
    merged_cfg = dict(merged_cfg)
    merged_cfg["normalize"] = True
    merged_cfg["device"] = "cpu"  # CPU generation runs on host; GPU path overrides

    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)
    states_shape, actions_shape, locations_shape = _shapes_from_config(config)
    shapes = (states_shape, actions_shape, locations_shape)

    os.makedirs(data_dir, exist_ok=True)

    # Preallocate (sparse) files up front so workers can write disjoint ranges.
    _open_memmaps(
        data_dir, num_samples, states_shape, actions_shape, locations_shape, mode="w+"
    )

    bytes_states = num_samples * int(np.prod(states_shape)) * 2
    print(
        f"Generating {num_samples} samples (backend={backend}) -> {data_dir}\n"
        f"  states {(num_samples, *states_shape)} float16 "
        f"(~{bytes_states / 1e9:.1f} GB)",
        flush=True,
    )

    t0 = time.time()
    if backend == "cpu":
        chunk = max(1, math.ceil(num_samples / (num_workers * 8)))
        _generate_cpu(
            data_dir, num_samples, merged_cfg, shapes, num_workers, seed, chunk
        )
    elif backend == "gpu":
        merged_cfg_gpu = dict(merged_cfg)
        merged_cfg_gpu["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        _generate_gpu(
            data_dir, num_samples, merged_cfg_gpu, shapes, gen_batch_size, seed
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'cpu' or 'gpu')")

    meta = {
        "num_samples": num_samples,
        "backend": backend,
        "seed": seed,
        "states_shape": list(states_shape),
        "actions_shape": list(actions_shape),
        "locations_shape": list(locations_shape),
        "states_dtype": "float16",
        "arr_dtype": "float32",
        "img_size": config.img_size,
        "sample_length": config.sample_length,
        "config": merged_cfg,
    }
    with open(os.path.join(data_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    print(
        f"Done: {num_samples} samples in {dt:.0f}s "
        f"({num_samples / max(dt, 1e-6):.0f}/s). Meta -> {META_FILE}",
        flush=True,
    )


def _parse_args():
    p = argparse.ArgumentParser(description="Pre-generate the two-rooms dataset.")
    p.add_argument("--data-dir", required=True, help="Output directory for the dataset.")
    p.add_argument("--num-samples", type=int, default=1_200_000)
    p.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--num-workers", type=int, default=16, help="CPU backend only.")
    p.add_argument("--gen-batch-size", type=int, default=2048, help="GPU backend only.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    generate_and_save(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        backend=args.backend,
        num_workers=args.num_workers,
        gen_batch_size=args.gen_batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
