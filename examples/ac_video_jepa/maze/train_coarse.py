"""Train the LEVEL-1 coarse JEPA (psi + P_high) on the FROZEN fine world model.

Strict 2-level H-JEPA. The coarse level is trained by WM IMAGINATION ("dreams"):
for each start state z_t and each macro-option o in {down,up,right,left}, dream a
k-step latent rollout under o (`dream_macro_option`, wall-aware, stays in latent),
encode it with the EMA target psi to get the target coarse state, and train P_high
to predict it. VICReg (std + covariance) prevents collapse. WM is frozen.

Run: python -m examples.ac_video_jepa.maze.train_coarse <fine_ckpt> <out_dir> [epochs=8] [k=5]
Output: <out_dir>/coarse.pth
"""
import copy
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW

from eb_jepa.datasets.utils import init_data
from eb_jepa.hjepa import (CoarseEncoder, CoarsePredictor, N_OPTIONS,
                           coarse_jepa_loss, dream_macro_option, ema_update)
from eb_jepa.losses import CovarianceLoss, HingeStdLoss
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def main():
    fine_ckpt, out_dir = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5            # macro-option horizon (fine steps)
    coarse_dim = 128
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    cfg.data.batch_size = 256

    loader, _, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device)
    if data_pipeline is not None:
        data_pipeline.warm_up()
    cell_size = float(data_config.cell_size)

    jepa, f = build_fine(cfg, data_config, device)
    load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None, device=device, strict=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    psi = CoarseEncoder(in_dim=f, coarse_dim=coarse_dim).to(device)
    psi_ema = copy.deepcopy(psi).to(device)
    for p in psi_ema.parameters():
        p.requires_grad_(False)
    p_high = CoarsePredictor(coarse_dim=coarse_dim, n_options=N_OPTIONS).to(device)
    opt = AdamW(list(psi.parameters()) + list(p_high.parameters()), lr=1e-3, weight_decay=1e-5)
    std_fn, cov_fn = HingeStdLoss(), CovarianceLoss()
    print(f"[coarse] f={f} coarse_dim={coarse_dim} k={k} epochs={epochs} | strict 2-level H-JEPA", flush=True)

    for epoch in range(epochs):
        t0 = time.time(); tot = tp = tr = 0.0; nb = 0
        for x, a, loc, _, _ in loader:
            x = x.to(device, non_blocking=True).float()
            t = x.shape[2] // 2                                 # a mid-trajectory start frame
            obs0 = x[:, :, t:t + 1]                             # [B, C, 1, H, W]
            with torch.no_grad():
                z0 = jepa.encode(obs0)                          # [B, f, 1, 1, 1]  (frozen)
                s_targets = torch.stack(
                    [psi_ema(dream_macro_option(jepa, z0, o, k, cell_size)) for o in range(N_OPTIONS)],
                    dim=1).detach()                             # [B, n_opt, coarse_dim]
            loss, pred, reg = coarse_jepa_loss(psi, p_high, s_targets, z0, std_fn, cov_fn)
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(psi_ema, psi, tau=0.99)
            tot += loss.item(); tp += float(pred); tr += float(reg); nb += 1
        print(f"[coarse] epoch {epoch} {time.time()-t0:.0f}s loss={tot/max(nb,1):.4f} "
              f"pred={tp/max(nb,1):.4f} reg={tr/max(nb,1):.4f}", flush=True)
        torch.save({"psi": psi.state_dict(), "p_high": p_high.state_dict(),
                    "coarse_dim": coarse_dim, "k": k, "f": f},
                   os.path.join(out_dir, "coarse.pth"))

    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[coarse] DONE -> {out_dir}/coarse.pth", flush=True)


if __name__ == "__main__":
    main()
