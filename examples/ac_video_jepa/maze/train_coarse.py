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
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.optim import AdamW

from eb_jepa.datasets.utils import init_data
from eb_jepa.hjepa import (CoarseEncoder, CoarseDistanceHead, CoarsePredictor, N_OPTIONS,
                           coarse_jepa_loss, dream_macro_option, ema_update)
from eb_jepa.losses import CovarianceLoss, HingeStdLoss
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def main():
    fine_ckpt, out_dir = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5            # macro-option horizon (fine steps)
    coarse_dim = 32                                             # small -> cleaner Euclidean distance
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    cfg.data.sample_length = int(cfg.data.get("n_steps", 91)) - 1   # FULL path -> long-range distances
    cfg.data.batch_size = 128

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

    psi = CoarseEncoder(in_dim=f, coarse_dim=coarse_dim, layer_norm=True).to(device)
    psi_ema = copy.deepcopy(psi).to(device)
    for p in psi_ema.parameters():
        p.requires_grad_(False)
    p_high = CoarsePredictor(coarse_dim=coarse_dim, n_options=N_OPTIONS).to(device)
    d_head = CoarseDistanceHead(coarse_dim=coarse_dim).to(device)   # quasimetric on coarse states
    opt = AdamW(list(psi.parameters()) + list(p_high.parameters()) + list(d_head.parameters()),
                lr=1e-3, weight_decay=1e-5)
    std_fn, cov_fn = HingeStdLoss(), CovarianceLoss()
    wandb.init(project="eb_jepa", name=f"hjepa-coarse-k{k}", group="hjepa",
               config={"f": f, "coarse_dim": coarse_dim, "k": k, "epochs": epochs,
                       "lr": 1e-3, "n_options": N_OPTIONS, "ema_tau": 0.99,
                       "std_coeff": 16.0, "cov_coeff": 8.0})
    print(f"[coarse] f={f} coarse_dim={coarse_dim} k={k} epochs={epochs} | strict 2-level H-JEPA", flush=True)

    dist_coeff = 0.1     # temporal-distance regression: ||psi(z_i)-psi(z_j)|| -> |i-j| (steps)
    n_pairs = 256
    for epoch in range(epochs):
        t0 = time.time(); tot = tp = tr = td = tss = 0.0; nb = 0
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
            # (2) temporal-distance REGRESSION over the FULL trajectory: coarse distance ~ steps
            s_all = psi(z_all.permute(0, 2, 1, 3, 4).reshape(B * T, f, 1, 1, 1)).reshape(B, T, -1)  # [B,T,dc]
            ii = torch.randint(0, T, (n_pairs,), device=device)
            jj = torch.randint(0, T, (n_pairs,), device=device)
            si = s_all[:, ii].reshape(B * n_pairs, -1)
            sj = s_all[:, jj].reshape(B * n_pairs, -1)
            d_pred = d_head(si, sj).reshape(B, n_pairs)                         # learned quasimetric
            d_tgt = (ii - jj).abs().float().unsqueeze(0).expand(B, -1)          # [B, n_pairs]
            # CROSS-TRAJECTORY HARD NEGATIVES: every trajectory in the batch is a DIFFERENT maze,
            # so a state from another maze is unreachable -> its quasimetric distance must be large
            # (>= T). This is the missing REPULSIVE signal: without it the metric only learns the
            # within-path temporal order and lets walled-off/far regions read as "near the goal".
            if B > 1:
                perm = torch.roll(torch.arange(B, device=device), 1)            # b -> another maze
                kk = torch.randint(0, T, (n_pairs,), device=device)
                sn = s_all[perm][:, kk].reshape(B * n_pairs, -1)
                d_neg = d_head(si, sn).reshape(B, n_pairs)
                neg_loss = F.relu(float(T) - d_neg).mean()                      # hinge: push d >= T
            else:
                neg_loss = d_pred.new_zeros(())
            dist_loss = F.smooth_l1_loss(d_pred, d_tgt) + 0.25 * neg_loss
            loss = loss_pr + dist_coeff * dist_loss
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(psi_ema, psi, tau=0.99)
            tot += loss.item(); tp += float(pred); tr += float(reg)
            td += float(dist_loss); tss += s_all.std().item(); nb += 1
        nb = max(nb, 1)
        wandb.log({"coarse/loss": tot / nb, "coarse/pred": tp / nb, "coarse/reg": tr / nb,
                   "coarse/dist": td / nb, "coarse/s_std": tss / nb, "epoch": epoch})
        print(f"[coarse] epoch {epoch} {time.time()-t0:.0f}s loss={tot/nb:.4f} "
              f"pred={tp/nb:.4f} reg={tr/nb:.4f} dist={td/nb:.4f} s_std={tss/nb:.4f}", flush=True)
        torch.save({"psi": psi.state_dict(), "p_high": p_high.state_dict(),
                    "d_head": d_head.state_dict(), "layer_norm": True,
                    "coarse_dim": coarse_dim, "k": k, "f": f},
                   os.path.join(out_dir, "coarse.pth"))

    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[coarse] DONE -> {out_dir}/coarse.pth", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
