"""CPU maze dataset.

Each ``__getitem__`` call:
  1. Generates a random DFS maze (retried until A* path >= min_path_length).
  2. Solves A* from corner (1,1) to corner (H-2, W-2).
  3. Pads or truncates to ``n_steps`` (pad = stay at goal with zero action).
  4. Samples a random window of length ``sample_length``.
  5. Renders 2-channel frames (agent dot + maze walls).
  6. Normalises and returns a ``MazeSample``.

The 5-field NamedTuple (states / actions / locations / wall_x / door_y) matches
``WallSample`` so the same training loop unpacks both. ``wall_x`` and ``door_y``
are dummies (set to 0) since maze has no global wall layout.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import torch

from eb_jepa.datasets.maze.maze_generator import generate_maze
from eb_jepa.datasets.maze.maze_solver import (
    DIRECTIONS,
    solve_a_star,
)
from eb_jepa.datasets.maze.normalizer import MazeNormalizer


class MazeSample(NamedTuple):
    states: torch.Tensor     # (2, T, H, W) or (B, 2, T, H, W)
    actions: torch.Tensor    # (2, T) or (B, 2, T)
    locations: torch.Tensor  # (2, T) or (B, 2, T)
    wall_x: torch.Tensor     # dummy
    door_y: torch.Tensor     # dummy


@dataclass
class MazeDatasetConfig:
    # Maze geometry
    maze_height: int = 21
    maze_width: int = 21
    cell_size: int = 3
    img_size: int = 63  # = maze_height * cell_size

    # Path constraints
    n_steps: int = 91
    sample_length: int = 17
    min_path_length: int = 18
    max_gen_retries: int = 64
    # Fraction of trajectory steps replaced by a "wall bump": a cardinal action
    # INTO a wall (position unchanged). A* paths never bump walls, so without
    # this the world-model never learns collisions → at planning it predicts
    # movement for wall-directed actions → MPC drives into walls. 0 = off.
    wall_bump_prob: float = 0.0

    # Visualisation
    agent_std: float = 1.2

    # Dataset
    size: int = 100000
    val_size: int = 10000
    batch_size: int = 64
    train: bool = True
    device: str = "cpu"
    normalize: bool = True
    num_workers: int = 0
    pin_mem: bool = False
    persistent_workers: bool = False


# ---------------------------------------------------------------------------
# Static rendering helpers — pure torch CPU, no maze-state dependency.
# ---------------------------------------------------------------------------


def render_wall_mask(maze_grid: torch.Tensor, cell_size: int) -> torch.Tensor:
    """Upsample (H, W) discrete grid to (img_h, img_w) uint8 wall mask.

    maze_grid: 0 = wall, 1 = path. Returns 255 where wall, 0 where path.
    """
    # 1 - maze gives wall mask; repeat each cell cell_size times
    walls = (1 - maze_grid).to(torch.uint8) * 255
    return walls.repeat_interleave(cell_size, dim=-2).repeat_interleave(
        cell_size, dim=-1
    )


def render_dot(
    positions: torch.Tensor,
    img_size: int,
    agent_std: float,
    device=None,
) -> torch.Tensor:
    """Render a Gaussian dot at each position.

    positions: (..., 2) — pixel-space (row, col). May be batched and time-indexed.

    Returns uint8 tensor (..., img_size, img_size).
    """
    if device is None:
        device = positions.device
    lin = torch.arange(img_size, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([rr, cc], dim=-1)  # (H, W, 2)

    lead = positions.shape[:-1]
    grid = grid.view(*([1] * len(lead)), img_size, img_size, 2).expand(
        *lead, img_size, img_size, 2
    )
    pos = positions.unsqueeze(-2).unsqueeze(-2)  # (..., 1, 1, 2)
    d2 = (grid - pos).pow(2).sum(dim=-1)
    img = torch.exp(-d2 / (2.0 * agent_std * agent_std)) * 255.0
    return img.clamp(0, 255).to(torch.uint8)


def cell_to_pixel(cell_rc: np.ndarray, cell_size: int) -> np.ndarray:
    """Convert (..., 2) cell coords to pixel-space cell centres."""
    offset = (cell_size - 1) / 2.0
    return cell_rc.astype(np.float32) * cell_size + offset


# ---------------------------------------------------------------------------
# Sample generation (numpy/CPU; render delegated to torch helpers above).
# ---------------------------------------------------------------------------


def generate_path_and_actions(
    config: MazeDatasetConfig,
    rng: Optional[np.random.Generator] = None,
):
    """Generate one maze + A* path + padded trajectory.

    Returns:
        maze_grid: (H, W) uint8 numpy array (0=wall, 1=path)
        cell_positions: (n_steps, 2) int32 numpy array — cell coords over time
        action_vecs:    (n_steps - 1, 2) float32 numpy array — pixel-space displacements
        start_cell:     (2,)
        goal_cell:      (2,)
    """
    H, W = config.maze_height, config.maze_width
    start = (1, 1)
    goal = (H - 2, W - 2)
    sample_length = config.sample_length
    min_required = max(sample_length + 1, config.min_path_length)

    for _ in range(config.max_gen_retries):
        maze = generate_maze(H, W, rng=rng)
        # Ensure start and goal are on the path
        maze[start[0], start[1]] = 1
        maze[goal[0], goal[1]] = 1
        sol = solve_a_star(maze, start, goal)
        if sol is None:
            continue
        path, action_ids = sol  # path: list of (r, c); len=path_len; actions len=path_len-1
        if len(path) >= min_required:
            break
    else:
        # Fallback: accept whatever we got
        path, action_ids = sol if sol is not None else ([start], [])

    n_steps = config.n_steps
    n_act = n_steps - 1
    cs = config.cell_size

    cell_positions = np.zeros((n_steps, 2), dtype=np.int32)
    action_vecs = np.zeros((n_act, 2), dtype=np.float32)

    bump_prob = getattr(config, "wall_bump_prob", 0.0)
    if bump_prob > 0.0 and len(path) >= 2:
        # Interleave the A* path with "wall bumps": at each step, with prob
        # bump_prob, emit a cardinal action INTO a wall (position unchanged) so
        # the world-model learns collisions; otherwise advance along the path.
        _rng = rng if rng is not None else np.random.default_rng()
        cur = np.array(path[0], dtype=np.int32)
        cell_positions[0] = cur
        i = 0  # current path index (cur == path[i])
        for t in range(n_act):
            wall_dirs = [
                (dr, dc)
                for (dr, dc) in DIRECTIONS
                if not (
                    0 <= cur[0] + dr < H
                    and 0 <= cur[1] + dc < W
                    and maze[cur[0] + dr, cur[1] + dc] == 1
                )
            ]
            if wall_dirs and _rng.random() < bump_prob:
                dr, dc = wall_dirs[_rng.integers(len(wall_dirs))]
                action_vecs[t] = np.array([dr, dc], dtype=np.float32) * cs
                cell_positions[t + 1] = cur  # bumped a wall → stay
            elif i < len(path) - 1:
                nxt = np.array(path[i + 1], dtype=np.int32)
                action_vecs[t] = (nxt - cur).astype(np.float32) * cs
                cur = nxt
                i += 1
                cell_positions[t + 1] = cur
            else:
                cell_positions[t + 1] = cur  # reached goal → stay (zero action)
        return (
            maze,
            cell_positions,
            action_vecs,
            np.array(start, dtype=np.int32),
            np.array(goal, dtype=np.int32),
        )

    real_len = min(len(path), n_steps)
    cell_positions[:real_len] = np.array(path[:real_len], dtype=np.int32)
    # Pad: stay at last reached cell
    if real_len < n_steps:
        cell_positions[real_len:] = cell_positions[real_len - 1]

    n_real_actions = min(len(action_ids), n_act)
    if n_real_actions > 0:
        dirs = DIRECTIONS[np.array(action_ids[:n_real_actions], dtype=np.int64)]
        action_vecs[:n_real_actions] = dirs.astype(np.float32) * config.cell_size

    return (
        maze,
        cell_positions,
        action_vecs,
        np.array(start, dtype=np.int32),
        np.array(goal, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MazeDataset(torch.utils.data.Dataset):
    """Online maze dataset — generates one (maze, trajectory) per __getitem__."""

    def __init__(self, config: MazeDatasetConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        if config.normalize:
            self.normalizer = MazeNormalizer(img_size=config.img_size)
        else:
            self.normalizer = None
        # Worker-local rng; PyTorch's worker_init_seed will reseed numpy if
        # users follow standard practice. Default Generator is fine.
        self._rng = np.random.default_rng()

    def __len__(self):
        return self.config.size

    def __getitem__(self, idx):
        sample = self.generate_multistep_sample()
        # Squeeze the implicit batch dim used by precomputed.py pool consumers
        return sample._replace(
            states=sample.states.squeeze(0),
            actions=sample.actions.squeeze(0),
            locations=sample.locations.squeeze(0),
        )

    # Used by both __getitem__ (after squeeze) and the CPU stream pipeline.
    # Returns tensors with an explicit batch dim of size 1 to match
    # WallDataset.generate_multistep_sample() convention.
    def generate_multistep_sample(self) -> MazeSample:
        cfg = self.config
        maze, cell_positions, action_vecs, _, _ = generate_path_and_actions(
            cfg, rng=self._rng
        )

        # Convert to pixel-space cell centres
        pixel_positions = cell_to_pixel(cell_positions, cfg.cell_size)  # (T, 2)

        # Build tensors. All on CPU; normalisation/dtype handled below.
        positions_t = torch.from_numpy(pixel_positions)  # (T, 2) float32
        actions_t = torch.from_numpy(action_vecs)        # (T-1, 2) float32
        maze_t = torch.from_numpy(maze.astype(np.int64))  # (H, W) int64

        # Drop last frame so frames align with actions (T-1 of each).
        positions_t = positions_t[:-1]  # (T-1, 2)

        # Render
        T = positions_t.shape[0]
        wall_mask = render_wall_mask(maze_t, cfg.cell_size)  # (img, img) uint8
        wall_mask_t = wall_mask.unsqueeze(0).expand(T, -1, -1)  # (T, img, img)
        agent_imgs = render_dot(
            positions_t, cfg.img_size, cfg.agent_std, device=torch.device("cpu")
        )  # (T, img, img)

        # Stack channels: (T, 2, img, img)
        states = torch.stack([agent_imgs, wall_mask_t], dim=1).float()

        # Window sample
        sl = cfg.sample_length
        max_start = (cfg.n_steps - 1) - sl  # inclusive upper bound
        start = int(self._rng.integers(0, max(1, max_start + 1)))
        states = states[start : start + sl]      # (sl, 2, img, img)
        actions_t = actions_t[start : start + sl]  # (sl, 2)
        positions_t = positions_t[start : start + sl]  # (sl, 2)

        if cfg.normalize and self.normalizer is not None:
            states = self.normalizer.normalize_state(states)
            positions_t = self.normalizer.normalize_location(positions_t)

        # Reshape to eb_jepa convention: (B=1, C=2, T, H, W) for states,
        # (B=1, 2, T) for actions and locations.
        states = states.permute(1, 0, 2, 3).unsqueeze(0)        # (1, 2, T, H, W)
        actions_t = actions_t.permute(1, 0).unsqueeze(0)         # (1, 2, T)
        positions_t = positions_t.permute(1, 0).unsqueeze(0)     # (1, 2, T)

        return MazeSample(
            states=states,
            actions=actions_t,
            locations=positions_t,
            wall_x=torch.zeros(1),
            door_y=torch.zeros(1),
        )
