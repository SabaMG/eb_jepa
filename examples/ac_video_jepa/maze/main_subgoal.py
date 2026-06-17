"""Train the high-level SUBGOAL predictor (learned replacement for A* waypoints).

Feudal/closed-loop hierarchy: SubgoalPredictor(z_current, goal_xy) -> position of
the next waypoint ~N cells along the route to the goal. Supervised on A*
trajectories (label = the A* position N frames ahead). The fine world model +
encoder + probe are frozen. At eval (eval_subgoal.py) the predictor proposes the
waypoints and a low-level reacher follows them — NO A* at eval.

Run: python -m examples.ac_video_jepa.maze.main_subgoal <fine_ckpt> <out_dir> [N=4] [epochs=12]
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW

from eb_jepa.datasets.utils import init_data
from eb_jepa.hierarchical import SubgoalPredictor
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def main():
    fine_ckpt, out_dir = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    cfg.data.sample_length = int(cfg.data.get("n_steps", 91)) - 1  # long: far goals
    cfg.data.batch_size = 96

    loader, _, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device)

    jepa, f = build_fine(cfg, data_config, device)
    info = load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None,
                           device=device, strict=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    subgoal = SubgoalPredictor(f).to(device)
    opt = AdamW(subgoal.parameters(), lr=1e-3, weight_decay=1e-5)
    print(f"[subgoal] f={f} N={N} epochs={epochs} | predicts the next A* waypoint", flush=True)

    for epoch in range(epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        for x, a, loc, _, _ in loader:
            x = x.to(device, non_blocking=True)
            loc = loc.to(device, non_blocking=True)             # [B,2,T] normalized positions
            B, _, T = loc.shape
            with torch.no_grad():
                z = jepa.encode(x)                              # [B,f,T,1,1]
            z_flat = z.permute(0, 2, 1, 3, 4).reshape(B * T, f)  # [B*T,f]
            goal = loc[:, :, -1:].expand(B, 2, T).permute(0, 2, 1).reshape(B * T, 2)
            idx = torch.clamp(torch.arange(T, device=device) + N, max=T - 1)
            label = loc[:, :, idx].permute(0, 2, 1).reshape(B * T, 2)
            pred = subgoal(z_flat, goal)
            loss = F.mse_loss(pred, label)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"[subgoal] epoch {epoch} {time.time()-t0:.0f}s mse={tot/max(nb,1):.5f}", flush=True)
        torch.save({"subgoal": subgoal.state_dict(), "N": N, "f": f},
                   os.path.join(out_dir, "subgoal.pth.tar"))
    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[subgoal] DONE -> {out_dir}/subgoal.pth.tar", flush=True)


if __name__ == "__main__":
    main()
