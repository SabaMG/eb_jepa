"""Random-walk baseline for the A*-free maze, SAME protocol as eval_subgoal.

Policy: keep moving in the current cardinal direction; only pick a NEW (random)
direction WHEN the move hits a wall (agent didn't move). Same difficulty-
proportional budget (factor*len(A*)+margin) and same metrics (success + SPL), so
it is directly comparable to the learned hierarchical agent.

Run: python -m examples.ac_video_jepa.maze.eval_random <fine_ckpt_for_cfg> <res> [num_ep=32] [factor=4] [margin=10] [seed=0]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.hierarchical import CARDINALS


def main():
    cfg_ckpt, rdir = sys.argv[1], sys.argv[2]
    num_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    factor = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    margin = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    os.makedirs(rdir, exist_ok=True)
    cfg = OmegaConf.load(Path(cfg_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True))
    cell_size = float(env_config.cell_size)
    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=800,
                     n_steps=800, max_step_norm=1.5)
    rng = np.random.default_rng(seed)
    print(f"[random-eval] random-persist baseline | budget {factor}xA*+{margin}", flush=True)

    successes, spls = [], []
    for ep in range(num_ep):
        env.reset()
        env.step(np.zeros(env.action_space.shape[0]))
        grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                              tuple(int(c) for c in env.goal_cell))
        astar_len = (len(solved[0]) - 1) if solved else 100
        max_steps = int(factor * astar_len + margin)
        cur = int(rng.integers(4)); n_moves = 0; success = False
        for _ in range(max_steps):
            prev = env.agent_cell.copy()
            _, _, done, trunc, _ = env.step((CARDINALS[cur] * cell_size).cpu().numpy())
            if not np.array_equal(env.agent_cell, prev):
                n_moves += 1
            else:
                # hit a wall -> pick a new random direction (different from current)
                cur = int(rng.choice([d for d in range(4) if d != cur]))
            if done:
                success = True; break
            if trunc:
                break
        successes.append(float(success))
        spls.append((astar_len / max(n_moves, astar_len)) if success else 0.0)
        print(f"[random-eval] ep {ep}: {'SUCCESS' if success else 'fail'} "
              f"(A*={astar_len} budget={max_steps} moves={n_moves})", flush=True)
    sr = float(np.mean(successes)); spl = float(np.mean(spls))
    json.dump({"policy": "random-persist", "success_rate": sr, "spl": spl,
               "num_episodes": num_ep, "budget": f"{factor}xA*+{margin}", "astar_free": True},
              open(os.path.join(rdir, "random_eval.json"), "w"), indent=2)
    print(f"[random-eval] RANDOM baseline: success={sr*100:.2f}%  SPL={spl:.3f}  over {num_ep} mazes", flush=True)


if __name__ == "__main__":
    main()
