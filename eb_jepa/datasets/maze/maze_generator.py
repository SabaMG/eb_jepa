"""Maze generation — random DFS (perfect maze).

Provides single-maze and batched generators. Both run on CPU with numpy: the
algorithm is branchy and stack-based, which does not vectorise well on GPU.
For the GPU streaming pipeline, we still generate mazes here on CPU then
upload the discrete grids to GPU for rendering.
"""

import random
from typing import List, Tuple

import numpy as np


def generate_maze(height: int, width: int, rng: np.random.Generator = None) -> np.ndarray:
    """Generate a perfect maze via randomised DFS.

    Args:
        height, width: Maze dimensions (must be odd).
        rng: Optional numpy RNG; if None uses ``random``.

    Returns:
        uint8 array (H, W): 0 = wall, 1 = path.
    """
    assert height % 2 == 1 and width % 2 == 1, (
        f"Maze dims must be odd, got {height}x{width}"
    )
    maze = np.zeros((height, width), dtype=np.uint8)
    start_r, start_c = 1, 1
    maze[start_r, start_c] = 1

    stack = [(start_r, start_c)]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    _choice = (rng.choice if rng is not None else random.choice)
    _shuffle = (rng.shuffle if rng is not None else random.shuffle)

    while stack:
        r, c = stack[-1]
        unvisited = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 < nr < height and 0 < nc < width and maze[nr, nc] == 0:
                unvisited.append((dr, dc))

        if unvisited:
            if rng is not None:
                # rng.choice with list-of-tuples returns ndarray; pick index instead
                idx = int(rng.integers(0, len(unvisited)))
                dr, dc = unvisited[idx]
            else:
                dr, dc = random.choice(unvisited)
            maze[r + dr // 2, c + dc // 2] = 1
            maze[r + dr, c + dc] = 1
            stack.append((r + dr, c + dc))
        else:
            stack.pop()
    return maze


def generate_maze_batch(
    batch_size: int,
    height: int,
    width: int,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate ``batch_size`` independent mazes.

    Returns:
        uint8 array (B, H, W).
    """
    out = np.empty((batch_size, height, width), dtype=np.uint8)
    for i in range(batch_size):
        out[i] = generate_maze(height, width, rng=rng)
    return out
