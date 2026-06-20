"""A*-FREE maze eval with a strict 2-level H-JEPA (TWO abstractions).

HIGH level (coarse, every m steps): from s_t = psi(z_t), `coarse_beam` plans over
macro-options with the coarse predictor P_high and returns the next coarse subgoal
s_sg. LOW level (fine, each step): pick the cardinal whose 1-step fine-WM prediction
lands closest to s_sg in coarse space. Two abstractions, two predictors, two time
scales. No A* in the decision loop; seeded; reports SPL.

Run: python -m examples.ac_video_jepa.maze.eval_hjepa \
        <fine_ckpt> <coarse_pth> <out_dir> <num_ep> <Hc> <m> <beam_W> <budget_factor> <margin> <seed>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.hjepa import CoarseEncoder, CoarsePredictor, coarse_beam, rank_fine_actions
from eb_jepa.hierarchical import CARDINALS
from eb_jepa.training_utils import load_checkpoint
from eb_jepa.vis_utils import save_gif
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


@torch.no_grad()
def main():
    a = sys.argv
    fine_ckpt, coarse_pth, rdir = a[1], a[2], a[3]
    num_ep = int(a[4]); Hc = int(a[5]); m = int(a[6]); beam_W = int(a[7])
    budget_factor = float(a[8]); margin = int(a[9]); seed = int(a[10]) if len(a) > 10 else 0
    n_gifs = int(a[11]) if len(a) > 11 else 0   # save a GIF of the first n_gifs episodes
    os.makedirs(rdir, exist_ok=True)
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
    psi = CoarseEncoder(in_dim=ck["f"], coarse_dim=ck["coarse_dim"]).to(device)
    psi.load_state_dict(ck["psi"]); psi.eval()
    p_high = CoarsePredictor(coarse_dim=ck["coarse_dim"]).to(device)
    p_high.load_state_dict(ck["p_high"]); p_high.eval()

    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=800,
                     max_step_norm=1.5, rng=np.random.default_rng(seed))
    norm = env.normalizer

    def enc(o):
        ot = norm.normalize_state(o.to(dtype=torch.float32, device=device)).unsqueeze(0).unsqueeze(2)
        return jepa.encode(ot)

    wandb.init(project="eb_jepa", name=f"hjepa-eval-seed{seed}", group="hjepa",
               config={"levels": 2, "num_ep": num_ep, "Hc": Hc, "m": m, "beam_W": beam_W,
                       "budget_factor": budget_factor, "margin": margin, "seed": seed,
                       "coarse_dim": ck["coarse_dim"], "k": ck["k"]})
    successes, spls = 0, []
    print(f"[hjepa-eval] Hc={Hc} m={m} W={beam_W} seed={seed} | 2-level | {num_ep} mazes", flush=True)
    for ep in range(num_ep):
        obs, info = env.reset()
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
        s_goal = psi(enc(info["target_obs"]))
        grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                              tuple(int(c) for c in env.goal_cell))
        astar_len = (len(solved[0]) - 1) if solved else 100
        budget = min(int(budget_factor * astar_len + margin), 800)
        goal_img = info["target_obs"]
        frames = [obs]
        s_sg = None; moves = 0; done = False
        blocked = {}; last_rev = -1
        OPP = {0: 1, 1: 0, 2: 3, 3: 2}    # opposite cardinal (no immediate U-turn)
        for step in range(budget):
            z_t = enc(obs)
            if step % m == 0 or s_sg is None:
                _o_star, s_sg = coarse_beam(p_high, psi(z_t), s_goal, Hc, beam_W)
            cell = tuple(int(c) for c in env.agent_cell)
            order = rank_fine_actions(jepa, psi, z_t, s_sg, cell_size)
            # try best-first, skipping blacklisted moves at this cell and the immediate U-turn
            cand = [d for d in order if d not in blocked.get(cell, set()) and d != last_rev]
            cand += [d for d in order if d not in cand]
            moved = False
            for d in cand:
                prev = env.agent_cell.copy()
                obs, _, done, trunc, info = env.step((CARDINALS[d] * cell_size).cpu().numpy())
                moves += 1
                if not np.array_equal(env.agent_cell, prev):
                    moved = True; last_rev = OPP[d]; frames.append(obs); break
                blocked.setdefault(cell, set()).add(d)   # didn't move -> wall -> blacklist
                if done or trunc:
                    break
            if done:
                break
            if not moved:        # fully boxed in (shouldn't happen) -> stop
                break
        if done:
            successes += 1; spls.append(astar_len / max(moves, astar_len))
        else:
            spls.append(0.0)
        if ep < n_gifs and len(frames) > 1:
            label = "succ" if done else "fail"
            save_gif(torch.stack([f.to(torch.float32) for f in frames]),
                     os.path.join(rdir, f"ep{ep}_{label}.gif"), fps=8,
                     show_frame_numbers=True, goal_frame=goal_img)
        wandb.log({"eval/success": float(done), "eval/spl": spls[-1],
                   "eval/astar_len": astar_len, "eval/moves": moves, "eval/ep": ep,
                   "eval/running_success_rate": successes / (ep + 1)})
        print(f"[hjepa-eval] ep {ep}: {'SUCCESS' if done else 'fail'}", flush=True)

    sr = successes / num_ep
    spl = float(np.mean(spls))
    wandb.run.summary["success_rate"] = sr
    wandb.run.summary["spl"] = spl
    wandb.log({"eval/success_rate": sr, "eval/spl_mean": spl})
    json.dump({"success_rate": sr, "spl": spl, "num_episodes": num_ep, "levels": 2,
               "Hc": Hc, "m": m, "beam_W": beam_W, "seed": seed, "astar_free": True},
              open(os.path.join(rdir, "hjepa_eval.json"), "w"), indent=2)
    print(f"[hjepa-eval] A*-FREE 2-level success={sr*100:.2f}%  SPL={spl:.3f}  over {num_ep} mazes (seed {seed})", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
