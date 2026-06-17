"""Top-level dataset dispatcher.

``init_data`` is the single entry point used by ``examples/ac_video_jepa/main.py``.
It selects an environment (``two_rooms`` or ``maze``) and within it a pipeline
mode (``online`` / ``stream`` / ``offline``). Stream mode further selects a
backend (``cpu`` workers or ``gpu`` vectorised generator).

Each environment lives in its own sub-package; this dispatcher only knows
their entry points.
"""

from pathlib import Path

import torch
import yaml

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml

DATASETS_DIR = Path(__file__).parent

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_env_data_config(env_name: str, overrides: dict = None) -> dict:
    """Load base data config for an environment and apply overrides."""
    config_path = DATASETS_DIR / env_name / "data_config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    if overrides:
        base_config.update(overrides)
    return base_config


def _resolve_env(env_name):
    """Return ``(DatasetClass, ConfigClass, Normalizer)`` for env_name."""
    if env_name == "two_rooms":
        from eb_jepa.datasets.two_rooms.normalizer import Normalizer
        from eb_jepa.datasets.two_rooms.wall_dataset import (
            WallDataset,
            WallDatasetConfig,
        )
        return WallDataset, WallDatasetConfig, Normalizer
    elif env_name == "maze":
        from eb_jepa.datasets.maze.maze_dataset import (
            MazeDataset,
            MazeDatasetConfig,
        )
        from eb_jepa.datasets.maze.normalizer import MazeNormalizer
        return MazeDataset, MazeDatasetConfig, MazeNormalizer
    else:
        raise ValueError(
            f"Unknown env_name={env_name!r}; expected 'two_rooms' or 'maze'"
        )


def create_env(env_name, config, **kwargs):
    """Build the gym-style env used by planning evaluation."""
    if env_name == "two_rooms":
        from eb_jepa.datasets.two_rooms.env import DotWall

        return DotWall(config=config, **kwargs)
    elif env_name == "maze":
        from eb_jepa.datasets.maze.env import MazeEnv

        return MazeEnv(config=config, **kwargs)
    else:
        raise ValueError(
            f"Unknown env_name={env_name!r}; expected 'two_rooms' or 'maze'"
        )


def _init_gpu_stream(env_name, merged_cfg, config, chunk_size, device, dtype,
                      gen_batch_size, num_gen_workers=0):
    """Dispatch to the env-specific GPU stream pipeline."""
    if env_name == "two_rooms":
        from eb_jepa.datasets.two_rooms.gpu_precomputed import init_gpu_precomputed_data

        return init_gpu_precomputed_data(
            env_config_dict=merged_cfg,
            chunk_size=chunk_size,
            epoch_size=config.size,
            batch_size=config.batch_size,
            device=device,
            dtype=dtype,
            gen_batch_size=gen_batch_size,
            drop_last=True,
        )
    elif env_name == "maze":
        from eb_jepa.datasets.maze.gpu_generator import init_gpu_maze_data

        return init_gpu_maze_data(
            config=config,
            chunk_size=chunk_size,
            epoch_size=config.size,
            batch_size=config.batch_size,
            device=device,
            dtype=dtype,
            gen_batch_size=gen_batch_size,
            num_workers=num_gen_workers,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unknown env_name={env_name!r}")


def _init_cpu_stream(env_name, merged_cfg, config, chunk_size, device, dtype,
                      num_gen_workers, normalizer):
    """Dispatch to the (env-agnostic) CPU stream pipeline."""
    from eb_jepa.datasets.precomputed import init_precomputed_data

    return init_precomputed_data(
        env_config_dict=merged_cfg,
        chunk_size=chunk_size,
        epoch_size=config.size,
        batch_size=config.batch_size,
        device=device,
        dtype=dtype,
        num_workers=num_gen_workers,
        drop_last=True,
        env_name=env_name,
        normalizer=normalizer,
    )


def _init_offline(env_name, pipeline_cfg, config, loader_kwargs, device=None):
    """Build the offline loader. Returns ``(loader, manager)``.

    ``manager`` is None for the naive random-access DataLoader path, and a
    non-None ``PipelineManager``/``DeepPrefetchManager`` when
    ``pipeline.stream=True`` (two_rooms only) — caller then owns its
    ``warm_up``/``shutdown`` lifecycle.
    """
    data_dir = pipeline_cfg.get("data_dir")
    if not data_dir:
        raise ValueError(
            "init_data: pipeline.data_dir must be set when pipeline.mode='offline'"
        )
    if env_name == "two_rooms":
        # pipeline.stream=True: read through the double-buffered VRAM stream
        # pipeline (large sequential chunk reads, no per-sample random access).
        # Much faster than the naive DataLoader on Lustre; traverses the dataset
        # in stored order (no shuffle by default).
        if bool(pipeline_cfg.get("stream", False)):
            if device is None:
                raise ValueError(
                    "init_data: device must be provided when pipeline.stream=True"
                )
            from eb_jepa.datasets.two_rooms.offline_dataset import (
                init_offline_stream_data,
            )

            chunk_size = int(pipeline_cfg.get("chunk_size", 9600))
            dtype_name = str(pipeline_cfg.get("dtype", "bfloat16")).lower()
            dtype = _DTYPE_MAP.get(dtype_name)
            if dtype is None:
                raise ValueError(
                    f"Unknown pipeline.dtype={dtype_name!r}; "
                    f"expected one of {list(_DTYPE_MAP)}"
                )
            shuffle = bool(pipeline_cfg.get("shuffle", False))
            # read_workers: intra-chunk parallel disk reads (>1 fans the read of
            # each chunk over disjoint sub-ranges — multiple outstanding I/O
            # requests against Lustre, the fix for slow single-threaded reads).
            # prefetch_depth: number of chunks kept reading+staging continuously
            # in VRAM (>1 -> DeepPrefetchManager; chunks load without blocking and
            # each GPU chunk is freed as soon as it is consumed).
            read_workers = int(pipeline_cfg.get("read_workers", 1))
            prefetch_depth = int(pipeline_cfg.get("prefetch_depth", 1))
            # epoch_size = config.size (per-epoch budget, e.g. 100k = 260 steps),
            # NOT the dataset size: one epoch matches the online baseline's budget,
            # and successive epochs advance through the dataset (chunk_id keeps
            # incrementing), so optim.epochs × size samples are consumed in order.
            loader, manager = init_offline_stream_data(
                data_dir=data_dir,
                chunk_size=chunk_size,
                epoch_size=config.size,
                batch_size=config.batch_size,
                device=device,
                dtype=dtype,
                drop_last=True,
                shuffle=shuffle,
                read_workers=read_workers,
                prefetch_depth=prefetch_depth,
            )
            return loader, manager

        from eb_jepa.datasets.two_rooms.offline_dataset import OfflineWallDataset

        dset = OfflineWallDataset(data_dir)
    elif env_name == "maze":
        raise NotImplementedError(
            "Offline mode is not yet supported for env_name='maze' — use 'online' or 'stream'"
        )
    else:
        raise ValueError(f"Unknown env_name={env_name!r}")
    config.size = len(dset)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=config.batch_size, shuffle=True, **loader_kwargs
    )
    return loader, None


def init_data(env_name, cfg_data=None, device=None, **kwargs):
    """Initialize data loaders for the specified environment.

    Supports three pipeline modes via ``cfg_data["pipeline"]["mode"]``:

      - ``online`` (default): standard DataLoader with on-the-fly CPU generation.
      - ``stream``: GPU-resident double-buffered pipeline; swaps a small chunk
        into VRAM every N training steps. ``pipeline.backend`` selects the
        generation backend: ``cpu`` (worker pool) or ``gpu`` (vectorised on GPU).
        Caller MUST invoke ``manager.warm_up()`` before iterating and
        ``manager.shutdown()`` at the end of training.
      - ``offline``: read pre-generated memmaps from disk. Only ``two_rooms``.

    For ``two_rooms`` offline, ``pipeline.stream=True`` reads the pre-generated
    dataset through the double-buffered VRAM stream pipeline (large sequential
    chunk reads) instead of a per-sample random-access DataLoader — far faster
    on Lustre — and returns a non-None manager (call ``warm_up``/``shutdown``).

    Supported envs: ``two_rooms``, ``maze``.

    Returns:
        Tuple of ``(train_loader, val_loader, config, pipeline_manager)``.
        ``pipeline_manager`` is None for ``online`` and ``offline`` modes.
    """
    DatasetClass, ConfigClass, NormalizerCls = _resolve_env(env_name)

    merged_cfg = load_env_data_config(env_name, cfg_data)
    config = update_config_from_yaml(ConfigClass, merged_cfg)

    pipeline_cfg = (cfg_data or {}).get("pipeline") or {}
    mode = str(pipeline_cfg.get("mode", "online")).lower()

    num_workers = merged_cfg.get("num_workers", 0)
    pin_mem = merged_cfg.get("pin_mem", False)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0
    prefetch_factor = merged_cfg.get("prefetch_factor")

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Validation loader: always a small online generator (never the bottleneck).
    val_dset = DatasetClass(config=config)
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=4, shuffle=False, **loader_kwargs
    )

    # ---- stream mode ----
    if mode == "stream":
        if device is None:
            raise ValueError(
                "init_data: device must be provided when pipeline.mode='stream'"
            )
        chunk_size = int(pipeline_cfg.get("chunk_size", merged_cfg["size"]))
        dtype_name = str(pipeline_cfg.get("dtype", "bfloat16")).lower()
        dtype = _DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(
                f"Unknown pipeline.dtype={dtype_name!r}; expected one of {list(_DTYPE_MAP)}"
            )
        backend = str(pipeline_cfg.get("backend", "cpu")).lower()

        if backend == "gpu":
            gen_batch_size = pipeline_cfg.get("gen_batch_size")
            gen_batch_size = int(gen_batch_size) if gen_batch_size else None
            num_gen_workers = int(pipeline_cfg.get("num_gen_workers", 0))
            loader, manager = _init_gpu_stream(
                env_name, merged_cfg, config, chunk_size, device, dtype,
                gen_batch_size, num_gen_workers,
            )
        elif backend == "cpu":
            num_gen_workers = int(pipeline_cfg.get("num_gen_workers", 16))
            # The PipelineLoader's _DatasetView needs a normalizer matching the env.
            if env_name == "maze":
                normalizer = NormalizerCls(img_size=config.img_size)
            else:
                normalizer = NormalizerCls()
            loader, manager = _init_cpu_stream(
                env_name, merged_cfg, config, chunk_size, device, dtype,
                num_gen_workers, normalizer,
            )
        else:
            raise ValueError(
                f"Unknown pipeline.backend={backend!r}; expected 'cpu' or 'gpu'"
            )
        return loader, val_loader, config, manager

    # ---- offline mode ----
    if mode == "offline":
        loader, manager = _init_offline(
            env_name, pipeline_cfg, config, loader_kwargs, device=device
        )
        return loader, val_loader, config, manager

    # ---- online mode (default) ----
    if mode != "online":
        raise ValueError(
            f"Unknown pipeline.mode={mode!r}; expected 'online', 'stream', or 'offline'"
        )
    dset = DatasetClass(config=config)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=config.batch_size, shuffle=True, **loader_kwargs
    )
    return loader, val_loader, config, None
