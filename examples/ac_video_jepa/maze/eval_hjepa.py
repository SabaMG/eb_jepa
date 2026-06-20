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
from eb_jepa.hjepa import (CoarseEncoder, CoarseDistanceHead, CoarsePredictor,
                           coarse_beam, dream_macro_option, dream_subgoal, rank_fine_actions)
from eb_jepa.hierarchical import CARDINALS
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def _maze_rgb(obs):
    """Maze background in the original style (green walls, black paths). obs: [C,H,W]."""
    a = obs.detach().cpu().numpy()
    img = np.zeros((a.shape[-2], a.shape[-1], 3), np.uint8)
    img[a[1] > 0.1] = (0, 200, 0)
    return img


def _centroid(mask):
    """(x, y) centre of a [H,W] dot mask, or None if empty."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0.2)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _save_annot_gif(frames, path, fps=8, scale=6, r=3):
    """frames = list of (maze_rgb, agent_xy, goal_xy, sg_xy). Upscale, draw small markers
    (goal yellow, subgoal cyan RING, agent red) + a frame counter — old-gif look."""
    from PIL import Image, ImageDraw
    n = len(frames)
    ims = []
    for i, (rgb, ac, gc, sc) in enumerate(frames):
        im = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)
        if gc:
            d.ellipse([gc[0] * scale - r, gc[1] * scale - r, gc[0] * scale + r, gc[1] * scale + r], fill=(255, 230, 0))
        if sc:
            d.ellipse([sc[0] * scale - r - 2, sc[1] * scale - r - 2, sc[0] * scale + r + 2, sc[1] * scale + r + 2],
                      outline=(0, 235, 235), width=2)      # subgoal = cyan ring (visible anywhere)
        if ac:
            d.ellipse([ac[0] * scale - r, ac[1] * scale - r, ac[0] * scale + r, ac[1] * scale + r], fill=(235, 30, 30))
        d.text((3, 2), f"Frame {i + 1}/{n}", fill=(255, 255, 255))
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=max(1, int(1000 / fps)), loop=0)


@torch.no_grad()
def main():
    a = sys.argv
    fine_ckpt, coarse_pth, rdir = a[1], a[2], a[3]
    num_ep = int(a[4]); Hc = int(a[5]); m = int(a[6]); beam_W = int(a[7])
    budget_factor = float(a[8]); margin = int(a[9]); seed = int(a[10]) if len(a) > 10 else 0
    n_gifs = int(a[11]) if len(a) > 11 else 0   # save a GIF of the first n_gifs episodes
    low_depth = int(a[12]) if len(a) > 12 else 1  # fine-level dream lookahead (1 = greedy)
    sg_horizon = int(a[13]) if len(a) > 13 else 0  # >0 = dream-subgoal mode (proper dynamic subgoal)
    os.makedirs(rdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
                                    device=device)
    cell_size = float(env_config.cell_size)

    jepa, f = build_fine(cfg, env_config, device)
    info = load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None, device=device, strict=False)
    jepa.eval()
    ck = torch.load(coarse_pth, map_location=device, weights_only=False)
    psi = CoarseEncoder(in_dim=ck["f"], coarse_dim=ck["coarse_dim"],
                        layer_norm=ck.get("layer_norm", False)).to(device)
    psi.load_state_dict(ck["psi"]); psi.eval()
    p_high = CoarsePredictor(coarse_dim=ck["coarse_dim"]).to(device)
    p_high.load_state_dict(ck["p_high"]); p_high.eval()
    dist_fn = None
    if "d_head" in ck:                                  # learned quasimetric (Upgrades 3+4)
        d_head = CoarseDistanceHead(coarse_dim=ck["coarse_dim"]).to(device)
        d_head.load_state_dict(ck["d_head"]); d_head.eval()
        dist_fn = lambda sa, sb: d_head(sa, sb)

    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=800,
                     max_step_norm=1.5, rng=np.random.default_rng(seed))
    norm = env.normalizer
    k_macro = int(ck.get("k", 5))

    # position probe (for decoding the subgoal latent -> a maze position for the GIF overlay)
    xy_head = MLPXYHead(input_shape=f, normalizer=norm).to(device)
    if isinstance(info, dict) and "xy_head_state_dict" in info:
        xy_head.load_state_dict(info["xy_head_state_dict"])
    xy_head.eval()

    img_sz = int(env_config.img_size)

    def pos_dot(z):
        """Decode a latent's probe position to an env-rendered dot mask (for the overlay)."""
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        xy = torch.clamp(xy, 0, img_sz - 1).to(env.target_position.device)   # match env render device
        return env._render_dot_at(xy)[0].detach().cpu().numpy()              # [H,W] dot mask

    def subgoal_dot(z_t, o_star):
        """Decode the chosen macro-option's dreamed endpoint to a maze-position dot mask."""
        return pos_dot(dream_macro_option(jepa, z_t, o_star, k_macro, cell_size))

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
        is_gif = ep < n_gifs
        gif_frames = []
        goal_c = _centroid(goal_img[0].detach().cpu().numpy()) if is_gif else None
        cur_sg_xy = None
        s_sg = None; moves = 0; done = False
        blocked = {}; last_rev = -1
        hold = 0; reach_eps = 2.0; max_hold = max(4, 2 * sg_horizon)   # subgoal commitment
        OPP = {0: 1, 1: 0, 2: 3, 3: 2}    # opposite cardinal (no immediate U-turn)
        for step in range(budget):
            z_t = enc(obs)
            s_t = psi(z_t)
            # COMMIT to a subgoal until it is REACHED (or a max hold) -> stable, no teleporting
            if sg_horizon > 0:
                d_to_sg = 1e9 if s_sg is None else float(
                    (dist_fn(s_t, s_sg) if dist_fn else torch.norm(s_t - s_sg, dim=-1))[0])
                replan = (s_sg is None) or (d_to_sg < reach_eps) or (hold >= max_hold)
            else:
                replan = (step % m == 0) or (s_sg is None)
            if replan:
                if sg_horizon > 0:
                    s_sg, z_sg = dream_subgoal(jepa, psi, z_t, s_goal, sg_horizon, beam_W,
                                               max(2, sg_horizon // 2), cell_size, dist_fn=dist_fn)
                    hold = 0
                    if is_gif:
                        cur_sg_xy = _centroid(pos_dot(z_sg))
                        agent_xy = _centroid(obs[0].detach().cpu().numpy())
                        print(f"   [ep{ep} s{step}] subgoal@{None if cur_sg_xy is None else (round(cur_sg_xy[0]),round(cur_sg_xy[1]))} "
                              f"agent@{None if agent_xy is None else (round(agent_xy[0]),round(agent_xy[1]))} "
                              f"goal@{None if goal_c is None else (round(goal_c[0]),round(goal_c[1]))}", flush=True)
                elif Hc == 0:
                    s_sg = s_goal
                    if is_gif:
                        cur_sg_xy = goal_c
                else:
                    _o_star, s_sg = coarse_beam(p_high, psi(z_t), s_goal, Hc, beam_W, dist_fn=dist_fn)
                    if is_gif:
                        cur_sg_xy = _centroid(subgoal_dot(z_t, _o_star))
            else:
                hold += 1
            cell = tuple(int(c) for c in env.agent_cell)
            order = rank_fine_actions(jepa, psi, z_t, s_sg, cell_size, depth=low_depth,
                                      width=beam_W, dist_fn=dist_fn)
            # try best-first, skipping blacklisted moves at this cell and the immediate U-turn
            cand = [d for d in order if d not in blocked.get(cell, set()) and d != last_rev]
            cand += [d for d in order if d not in cand]
            moved = False
            for d in cand:
                prev = env.agent_cell.copy()
                obs, _, done, trunc, info = env.step((CARDINALS[d] * cell_size).cpu().numpy())
                moves += 1
                if not np.array_equal(env.agent_cell, prev):
                    moved = True; last_rev = OPP[d]; frames.append(obs)
                    if is_gif:
                        gif_frames.append((_maze_rgb(obs), _centroid(obs[0].detach().cpu().numpy()),
                                           goal_c, cur_sg_xy))
                    break
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
        if is_gif and len(frames) > 1:
            label = "succ" if done else "fail"
            # annotated GIF: goal (yellow) + current subgoal (cyan) on every frame
            try:
                if len(gif_frames) > 1:
                    _save_annot_gif(gif_frames, os.path.join(rdir, f"ep{ep}_{label}_annot.gif"), fps=8)
            except Exception as e:
                print(f"[hjepa-eval] annot gif failed ep{ep}: {e}", flush=True)
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
               "Hc": Hc, "m": m, "beam_W": beam_W, "low_depth": low_depth, "seed": seed, "astar_free": True},
              open(os.path.join(rdir, "hjepa_eval.json"), "w"), indent=2)
    print(f"[hjepa-eval] A*-FREE 2-level success={sr*100:.2f}%  SPL={spl:.3f}  over {num_ep} mazes (seed {seed})", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
