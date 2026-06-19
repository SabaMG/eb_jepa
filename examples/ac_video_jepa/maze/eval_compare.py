"""Controlled comparison: FLAT vs HIERARCHICAL (global+local) maze navigation.

Same A*-free closed loop as eval_subgoal.py, but run in two modes on the same
mazes, to test the hypothesis: the global+local hierarchy removes the "door jitter"
(oscillation at narrow passages) of flat planning and is more efficient.

- mode "flat": the low-level reacher heads straight to the GLOBAL goal position
  (no subgoal). This is the no-hierarchy baseline (greedy-global).
- mode "hier": the SubgoalPredictor proposes a nearby waypoint and the reacher
  heads to it (the repo's Level 1).

Both modes share the exact same K-step lookahead reacher and anti-U-turn logic, so
the ONLY difference is the global vs global+local target. We report success, SPL,
and two jitter/efficiency proxies: revisits (cells entered more than once) and the
move/A* ratio.

Run: python -m examples.ac_video_jepa.maze.eval_compare <fine_ckpt> <subgoal_ckpt>
        <results_dir> [num_ep=32] [lookahead=4] [revisit_pen=0.05] [n_gifs=4]
        [budget_factor=4] [margin=10] [seed=0]
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
from eb_jepa.hierarchical import CARDINALS, SubgoalPredictor, fine_kstep_target
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from eb_jepa.vis_utils import save_gif
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine

OPP = {0: 1, 1: 0, 2: 3, 3: 2}


def run_pass(mode, num_ep, env, jepa, subgoal, xy_head, norm, cell_size, off,
             device, lookahead, revisit_pen, budget_factor, budget_margin,
             n_allowed, n_gifs, rdir, seed):
    """One eval pass over num_ep mazes in a given mode. Seeded so 'flat' and 'hier'
    see the same maze sequence (paired comparison)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def obs_tensor(o):
        return norm.normalize_state(o.to(dtype=torch.float32, device=device)).unsqueeze(0).unsqueeze(2)

    def probe_xy(z):
        return xy_head(z.float()).permute(0, 2, 1)[0, 0]

    def pred_cell(z):
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (int(round((float(xy[0]) - off) / cell_size)),
                int(round((float(xy[1]) - off) / cell_size)))

    successes, spls, revisits_l, moveratio_l = [], [], [], []
    for ep in range(num_ep):
        obs, info_e = env.reset()
        obs, _, _, _, info_e = env.step(np.zeros(env.action_space.shape[0]))
        goal_xy = norm.normalize_location(
            info_e["target_position"].to(dtype=torch.float32, device=device).unsqueeze(0))[0]
        goal_img = info_e["target_obs"] if "target_obs" in info_e else None
        grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                              tuple(int(c) for c in env.goal_cell))
        astar_len = (len(solved[0]) - 1) if solved else 100
        max_steps = min(int(budget_factor * astar_len + budget_margin), n_allowed)

        frames = [obs]; n_moves = 0; success = False
        blocked = {}; visit = {}; last_rev = -1
        for step in range(max_steps):
            ot = obs_tensor(obs)
            z = jepa.encode(ot)
            # --- the ONLY difference between the two modes ---
            if mode == "hier":
                sg = subgoal(z, goal_xy.unsqueeze(0))[0]   # learned nearby waypoint
            else:  # "flat": aim straight at the global goal (no hierarchy)
                sg = goal_xy
            cell = tuple(int(c) for c in env.agent_cell)
            visit[cell] = visit.get(cell, 0) + 1
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
        # jitter / efficiency metrics
        revisits = sum(v - 1 for v in visit.values())       # cells entered more than once
        move_ratio = n_moves / max(astar_len, 1)            # 1.0 = optimal-length path
        successes.append(float(success))
        spls.append((astar_len / max(n_moves, astar_len)) if success else 0.0)
        revisits_l.append(revisits)
        moveratio_l.append(move_ratio)
        if ep < n_gifs and len(frames) > 1:
            label = "succ" if success else "fail"
            try:
                save_gif(torch.stack([f.to(torch.float32) for f in frames]),
                         os.path.join(rdir, f"{mode}_ep{ep}_{label}.gif"), fps=8,
                         show_frame_numbers=True, goal_frame=goal_img)
            except Exception as e:
                print(f"   [gif {mode} ep{ep}] skipped: {e}", flush=True)
        print(f"[{mode}] ep {ep}: {'SUCCESS' if success else 'fail'} "
              f"moves={n_moves} revisits={revisits}", flush=True)
    return {
        "mode": mode,
        "success_rate": float(np.mean(successes)),
        "spl": float(np.mean(spls)),
        "mean_revisits": float(np.mean(revisits_l)),
        "mean_move_ratio": float(np.mean(moveratio_l)),
        "num_episodes": num_ep,
    }


@torch.no_grad()
def main():
    fine_ckpt, sg_ckpt, rdir = sys.argv[1], sys.argv[2], sys.argv[3]
    num_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    lookahead = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    revisit_pen = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
    n_gifs = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    budget_factor = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0
    budget_margin = int(sys.argv[9]) if len(sys.argv) > 9 else 10
    seed = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    os.makedirs(rdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True))
    cell_size = float(env_config.cell_size)
    off = (cell_size - 1) / 2.0
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

    print(f"[compare] flat vs hierarchical | {num_ep} mazes | lookahead={lookahead}", flush=True)
    results = {}
    for mode in ["flat", "hier"]:
        results[mode] = run_pass(mode, num_ep, env, jepa, subgoal, xy_head, norm,
                                 cell_size, off, device, lookahead, revisit_pen,
                                 budget_factor, budget_margin, n_allowed, n_gifs, rdir, seed)

    json.dump(results, open(os.path.join(rdir, "compare.json"), "w"), indent=2)
    print("\n=== FLAT vs HIERARCHICAL (global+local) ===", flush=True)
    print(f"{'metric':<16}{'flat':>10}{'hier':>10}", flush=True)
    for k in ["success_rate", "spl", "mean_revisits", "mean_move_ratio"]:
        print(f"{k:<16}{results['flat'][k]:>10.3f}{results['hier'][k]:>10.3f}", flush=True)
    print("\n-> hierarchy should show: higher success/SPL, fewer revisits, lower move ratio "
          "(less door jitter).", flush=True)


if __name__ == "__main__":
    main()
