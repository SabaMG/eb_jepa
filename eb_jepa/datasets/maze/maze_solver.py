"""A* maze solver — single + batched."""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

# Discrete action codes
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# Map (dr, dc) → action index
ACTION_MAP = {
    (-1, 0): ACTION_UP,
    (1, 0): ACTION_DOWN,
    (0, -1): ACTION_LEFT,
    (0, 1): ACTION_RIGHT,
}
# Inverse map: action index → unit (dr, dc) vector
DIRECTIONS = np.array(
    [
        [-1, 0],  # up
        [1, 0],   # down
        [0, -1],  # left
        [0, 1],   # right
    ],
    dtype=np.int32,
)


def _manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def solve_a_star(
    maze: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Optional[Tuple[List[Tuple[int, int]], List[int]]]:
    """A* shortest path on a 4-connected grid.

    Args:
        maze: uint8 (H, W). 1 = path, 0 = wall.
        start, end: (row, col).

    Returns:
        (path, actions) where path is a list of (r, c) and actions is a list of
        action indices of length len(path) - 1. None if no path exists.
    """
    height, width = maze.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            actions = []
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i + 1]
                actions.append(ACTION_MAP[(r2 - r1, c2 - c1)])
            return path, actions

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < height and 0 <= nc < width and maze[nr, nc] == 1:
                tentative = g_score[current] + 1
                neighbor = (nr, nc)
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f = tentative + _manhattan(neighbor, end)
                    heapq.heappush(open_set, (f, neighbor))
    return None
