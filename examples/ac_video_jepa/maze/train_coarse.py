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
import wandb
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
    wandb.init(project="eb_jepa", name=f"hjepa-coarse-k{k}", group="hjepa",
               config={"f": f, "coarse_dim": coarse_dim, "k": k, "epochs": epochs,
                       "lr": 1e-3, "n_options": N_OPTIONS, "ema_tau": 0.99,
                       "std_coeff": 16.0, "cov_coeff": 8.0})
    print(f"[coarse] f={f} coarse_dim={coarse_dim} k={k} epochs={epochs} | strict 2-level H-JEPA", flush=True)

    rank_coeff = 1.0     # temporal-distance shaping: closer in time -> closer in coarse space
    n_pairs = 128
    for epoch in range(epochs):
        t0 = time.time(); tot = tp = tr = tk = tss = 0.0; nb = 0
        for x, a, loc, _, _ in loader:
            x = x.to(device, non_blocking=True).float()
            B, _, T = x.shape[0], x.shape[1], x.shape[2]
            with torch.no_grad():
                z_all = jepa.encode(x)                         # [B, f, T, 1, 1]  (frozen)
                z0 = z_all[:, :, T // 2:T // 2 + 1]            # mid-trajectory start
                s_targets = torch.stack(
                    [psi_ema(dream_macro_option(jepa, z0, o, k, cell_size)) for o in range(N_OPTIONS)],
                    dim=1).detach()                             # [B, n_opt, coarse_dim]
            # (1) predictive (dream) + VICReg anti-collapse
            loss_pr, pred, reg = coarse_jepa_loss(psi, p_high, s_targets, z0, std_fn, cov_fn)
            # (2) temporal-distance RANKING: ||psi(z_i)-psi(z_j)|| ordered by |i-j| (scale-free)
            s_all = psi(z_all.permute(0, 2, 1, 3, 4).reshape(B * T, f, 1, 1, 1)).reshape(B, T, -1)  # [B,T,dc]
            ai = torch.randint(0, T, (n_pairs,), device=device)
            o1 = torch.randint(0, T, (n_pairs,), device=device)
            o2 = torch.randint(0, T, (n_pairs,), device=device)
            nearer = (ai - o1).abs() <= (ai - o2).abs()
            near = torch.where(nearer, o1, o2)
            far = torch.where(nearer, o2, o1)
            d_near = torch.norm(s_all[:, ai] - s_all[:, near], dim=-1)   # [B, n_pairs]
            d_far = torch.norm(s_all[:, ai] - s_all[:, far], dim=-1)
            rank_loss = torch.relu(1.0 + d_near - d_far).mean()
            loss = loss_pr + rank_coeff * rank_loss
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(psi_ema, psi, tau=0.99)
            tot += loss.item(); tp += float(pred); tr += float(reg)
            tk += float(rank_loss); tss += s_all.std().item(); nb += 1
        nb = max(nb, 1)
        wandb.log({"coarse/loss": tot / nb, "coarse/pred": tp / nb, "coarse/reg": tr / nb,
                   "coarse/rank": tk / nb, "coarse/s_std": tss / nb, "epoch": epoch})
        print(f"[coarse] epoch {epoch} {time.time()-t0:.0f}s loss={tot/nb:.4f} "
              f"pred={tp/nb:.4f} reg={tr/nb:.4f} rank={tk/nb:.4f} s_std={tss/nb:.4f}", flush=True)
        torch.save({"psi": psi.state_dict(), "p_high": p_high.state_dict(),
                    "coarse_dim": coarse_dim, "k": k, "f": f},
                   os.path.join(out_dir, "coarse.pth"))

    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[coarse] DONE -> {out_dir}/coarse.pth", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
