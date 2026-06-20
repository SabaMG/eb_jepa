"""Diagnostic for the 2-level H-JEPA: is the coarse space usable for navigation?

Walks the A* path of held-out mazes and measures:
  1. monotonic fraction of ||psi(z_t) - s_goal|| along the path (does coarse Euclidean
     distance decrease toward the goal?) -> tests the coarse METRIC.
  2. how often coarse_beam's chosen option matches the A* next move -> tests the coarse
     PLANNER.
If (1) is low, Euclidean coarse distance is not a navigation signal (need a coarse
quasimetric / distance head). If (1) is high but (2)/success is low, the bug is in the
planner/low-level.

Run: python -m examples.ac_video_jepa.maze.diag_hjepa <fine_ckpt> <coarse_pth> [num_ep=30] [Hc=4] [W=4]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.hierarchical import CARDINALS
from eb_jepa.hjepa import CoarseEncoder, CoarsePredictor, coarse_beam
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine

DELTA = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}   # cell delta -> CARDINALS index


@torch.no_grad()
def main():
    fine_ckpt, coarse_pth = sys.argv[1], sys.argv[2]
    num_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    Hc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    W = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
                                    device=device)
    cell_size = float(env_config.cell_size)
    jepa, f = build_fine(cfg, env_config, device)
    load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None, device=device, strict=False)
    jepa.eval()
    ck = torch.load(coarse_pth, map_location=device, weights_only=False)
    psi = CoarseEncoder(in_dim=ck["f"], coarse_dim=ck["coarse_dim"]).to(device); psi.load_state_dict(ck["psi"]); psi.eval()
    p_high = CoarsePredictor(coarse_dim=ck["coarse_dim"]).to(device); p_high.load_state_dict(ck["p_high"]); p_high.eval()
    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=800, max_step_norm=1.5,
                     rng=np.random.default_rng(0))
    norm = env.normalizer

    def enc(o):
        ot = norm.normalize_state(o.to(dtype=torch.float32, device=device)).unsqueeze(0).unsqueeze(2)
        return jepa.encode(ot)

    mono_fracs, beam_match = [], []
    for ep in range(num_ep):
        env.reset()
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
        s_goal = psi(enc(info["target_obs"]))
        grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                              tuple(int(c) for c in env.goal_cell))
        if not solved:
            continue
        cells = solved[0]
        dists, matches = [], []
        for idx, (prev, nxt) in enumerate(zip(cells[:-1], cells[1:])):
            s_t = psi(enc(obs))
            dists.append(float(torch.norm(s_t - s_goal)))
            o_astar = DELTA.get((nxt[0] - prev[0], nxt[1] - prev[1]))
            o_beam, _ = coarse_beam(p_high, s_t, s_goal, Hc, W)
            if o_astar is not None:
                matches.append(int(o_beam == o_astar))
                obs, _, _, _, info = env.step((CARDINALS[o_astar] * cell_size).cpu().numpy())
            else:
                break
        if len(dists) >= 2:
            d = torch.tensor(dists)
            mono_fracs.append(float((d[1:] < d[:-1]).float().mean()))
        if matches:
            beam_match.append(float(np.mean(matches)))

    print(f"[diag] coarse-distance monotonic fraction = {np.mean(mono_fracs):.3f} "
          f"(1.0 = ||s_t - s_goal|| always decreases toward goal)", flush=True)
    print(f"[diag] coarse_beam matches A* direction    = {np.mean(beam_match):.3f} "
          f"(1.0 = always picks the A* move)", flush=True)
    print(f"[diag] over {len(mono_fracs)} mazes. If metric frac is low (~0.5), the coarse "
          f"space needs a learned distance, not Euclidean.", flush=True)


if __name__ == "__main__":
    main()
