"""A*-FREE maze navigation with a LEARNED subgoal generator (no A* at eval).

Closed loop: at each step encode the obs, the SubgoalPredictor proposes the next
waypoint position, and a low-level reacher picks the cardinal whose FINE-world-model
1-step prediction (wall-aware) lands closest to that waypoint — with execution
feedback (blocked-direction skip). A* is used nowhere at eval.

Run: python -m examples.ac_video_jepa.maze.eval_subgoal <fine_ckpt> <subgoal_ckpt>
        <results_dir> [num_episodes=16]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.hierarchical import CARDINALS, SubgoalPredictor, fine_kstep_target
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from eb_jepa.vis_utils import save_gif
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine
from omegaconf import OmegaConf


@torch.no_grad()
def main():
    fine_ckpt, sg_ckpt, rdir = sys.argv[1], sys.argv[2], sys.argv[3]
    num_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    lookahead = int(sys.argv[5]) if len(sys.argv) > 5 else 1  # K-step fine-WM lookahead
    revisit_pen = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    n_gifs = int(sys.argv[7]) if len(sys.argv) > 7 else 0     # render GIFs for first n_gifs eps
    budget_factor = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0
    budget_margin = int(sys.argv[9]) if len(sys.argv) > 9 else 10
    os.makedirs(rdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True))
    cell_size = float(env_config.cell_size)
    # Per-episode step budget = factor * A* path length + margin. A* is used ONLY
    # to size the time limit (difficulty-proportional), NEVER for the agent's
    # decisions — navigation stays 100% A*-free. The env's hard cap is set high so
    # our per-episode budget (not the env) is the binding limit.
    n_allowed = 800

    jepa, f = build_fine(cfg, env_config, device)
    info = load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None,
                           device=device, strict=False)
    jepa.eval()
    sck = torch.load(sg_ckpt, map_location=device, weights_only=False)
    subgoal = SubgoalPredictor(f).to(device); subgoal.load_state_dict(sck["subgoal"]); subgoal.eval()

    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=n_allowed,
                     n_steps=n_allowed, max_step_norm=1.5)
    norm = env.normalizer
    xy_head = MLPXYHead(input_shape=f, normalizer=norm).to(device)
    if "xy_head_state_dict" in info:
        xy_head.load_state_dict(info["xy_head_state_dict"])
    xy_head.eval()
    print(f"[subgoal-eval] A*-FREE | N={sck['N']} | {num_ep} mazes", flush=True)

    def obs_tensor(o):
        return norm.normalize_state(o.to(dtype=torch.float32, device=device)).unsqueeze(0).unsqueeze(2)

    off = (cell_size - 1) / 2.0

    def probe_xy(z):  # -> [2] normalized
        return xy_head(z.float()).permute(0, 2, 1)[0, 0]

    def pred_cell(z):  # latent -> predicted maze cell via probe (pixel space)
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (int(round((float(xy[0]) - off) / cell_size)),
                int(round((float(xy[1]) - off) / cell_size)))

    print(f"[subgoal-eval] lookahead={lookahead} revisit_pen={revisit_pen} "
          f"budget={budget_factor}xA*+{budget_margin}", flush=True)
    successes = []; spls = []
    OPP = {0: 1, 1: 0, 2: 3, 3: 2}
    for ep in range(num_ep):
        obs, info_e = env.reset()
        obs, _, _, _, info_e = env.step(np.zeros(env.action_space.shape[0]))
        goal_xy = norm.normalize_location(
            info_e["target_position"].to(dtype=torch.float32, device=device).unsqueeze(0))[0]
        goal_img = info_e["target_obs"] if "target_obs" in info_e else None
        # A* path length (difficulty) -> per-episode step budget (A* only sets the
        # clock, never guides the agent).
        grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                              tuple(int(c) for c in env.goal_cell))
        astar_len = (len(solved[0]) - 1) if solved else 100
        max_steps = min(int(budget_factor * astar_len + budget_margin), n_allowed)
        frames = [obs]; n_moves = 0
        success = False; blocked = {}; visit = {}; last_rev = -1; verbose = (ep == 0)
        for step in range(max_steps):
            ot = obs_tensor(obs)
            z = jepa.encode(ot)
            sg = subgoal(z, goal_xy.unsqueeze(0))[0]            # [2] normalized waypoint
            cell = tuple(int(c) for c in env.agent_cell)
            visit[cell] = visit.get(cell, 0) + 1
            # score cardinals by a K-STEP fine-WM lookahead: roll the (wall-aware)
            # fine model K steps in that direction, distance of the endpoint to the
            # waypoint (+ revisit penalty). K-step lookahead avoids 1-step myopia
            # and dead-ends (a blocked dir's endpoint stays put -> far from waypoint).
            dist = []
            for dd in range(4):
                zf = fine_kstep_target(jepa, ot, torch.tensor([dd], device=device),
                                       lookahead, cell_size)
                d = float(torch.norm(probe_xy(zf) - sg).item())
                if revisit_pen > 0:
                    d += revisit_pen * visit.get(pred_cell(zf), 0)
                dist.append(d)
            order = sorted(range(4), key=lambda dd: dist[dd])
            cand = [d for d in order if d not in blocked.get(cell, set()) and d != last_rev]
            cand += [d for d in order if d not in cand]
            if verbose and step < 12:
                print(f"   [s{step}] cell={list(cell)} goal={env.goal_cell.tolist()} "
                      f"dist[D,U,R,L]={[round(x,2) for x in dist]}", flush=True)
            moved = False; done = False
            for d in cand:
                prev = env.agent_cell.copy()
                obs, _, done, trunc, info_e = env.step((CARDINALS[d] * cell_size).cpu().numpy())
                if not np.array_equal(env.agent_cell, prev):
                    moved = True; last_rev = OPP[d]; frames.append(obs); n_moves += 1; break
                blocked.setdefault(cell, set()).add(d)
                if done or trunc:
                    break
            if done:
                success = True; break
            if not moved:
                break
        successes.append(float(success))
        spls.append((astar_len / max(n_moves, astar_len)) if success else 0.0)
        if verbose:
            print(f"   ep0: A*_len={astar_len} budget={max_steps} moves={n_moves}", flush=True)
        if ep < n_gifs and len(frames) > 1:
            label = "succ" if success else "fail"
            try:
                save_gif(torch.stack([f.to(torch.float32) for f in frames]),
                         os.path.join(rdir, f"ep{ep}_{label}.gif"), fps=8,
                         show_frame_numbers=True, goal_frame=goal_img)
            except Exception as e:
                print(f"   [gif ep{ep}] skipped: {e}", flush=True)
        print(f"[subgoal-eval] ep {ep}: {'SUCCESS' if success else 'fail'}", flush=True)
    sr = float(np.mean(successes)); spl = float(np.mean(spls))
    json.dump({"success_rate": sr, "spl": spl, "num_episodes": num_ep, "N": sck["N"],
               "astar_free": True, "budget": f"{budget_factor}xA*+{budget_margin}"},
              open(os.path.join(rdir, "subgoal_eval.json"), "w"), indent=2)
    print(f"[subgoal-eval] A*-FREE success={sr*100:.2f}%  SPL={spl:.3f}  over {num_ep} mazes "
          f"(budget {budget_factor}xA*+{budget_margin})", flush=True)


if __name__ == "__main__":
    main()
