"""CO-TRAINING phase: jointly fine-tune the fine WM encoder/predictor together
with the high-level SubgoalPredictor (and the position probe), so the SHARED
latent is at once (a) dynamics-predictable, (b) position-decodable, (c) routable
for subgoal generation. Fixes the "non-shared bias" of the frozen baseline: the
encoder can now reshape its representation to serve the high level too.

Losses (all backprop into the shared encoder), low LR on enc/predictor:
  jepa_pred(+VC)   keep the wall-aware dynamics
  aux_pos          keep the latent position-decodable (eval uses the probe)
  subgoal          MSE to the A* waypoint N ahead (the routing signal)
The fine value is NOT used by the subgoal eval, so it is omitted here.

Run: python -m examples.ac_video_jepa.maze.main_cotrain <fine_ckpt> <subgoal_ckpt> <out_dir>
        [N=4] [epochs=8]
"""
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW

from eb_jepa.architectures import ImpalaEncoder, InverseDynamicsModel, RNNPredictor
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.hierarchical import SubgoalPredictor
from eb_jepa.training_utils import load_checkpoint, save_checkpoint
from eb_jepa.datasets.utils import init_data


def main():
    fine_ckpt, sg_ckpt, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    freeze_epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 3   # heads-only warmup
    enc_lr = float(sys.argv[7]) if len(sys.argv) > 7 else 5e-5      # gentle encoder LR
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    cfg.data.sample_length = int(cfg.data.get("n_steps", 91)) - 1
    cfg.data.batch_size = 96
    shutil.copy(Path(fine_ckpt).parent / "config.yaml", Path(out_dir) / "config.yaml")

    loader, _, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device)
    normalizer = loader.dataset.normalizer

    m = cfg.model
    enc = ImpalaEncoder(width=1, stack_sizes=(16, m.henc, m.dstc), num_blocks=2,
                        dropout_rate=None, layer_norm=False, input_channels=m.dobs,
                        final_ln=True, mlp_output_dim=512,
                        input_shape=(m.dobs, data_config.img_size, data_config.img_size))
    f = enc(torch.rand(1, m.dobs, 1, data_config.img_size, data_config.img_size)).shape[1]
    pred = RNNPredictor(hidden_size=enc.mlp_output_dim, final_ln=enc.final_ln)
    idm = InverseDynamicsModel(state_dim=f, hidden_dim=256, action_dim=2)
    reg = VC_IDM_Sim_Regularizer(
        cov_coeff=m.regularizer.cov_coeff, std_coeff=m.regularizer.std_coeff,
        sim_coeff_t=m.regularizer.sim_coeff_t, idm_coeff=m.regularizer.get("idm_coeff", 0.1),
        idm=idm, first_t_only=m.regularizer.get("first_t_only"), projector=None,
        spatial_as_samples=m.regularizer.spatial_as_samples,
        idm_after_proj=m.regularizer.idm_after_proj, sim_t_after_proj=m.regularizer.sim_t_after_proj)
    jepa = JEPA(enc, nn.Identity(), pred, reg, SquareLossSeq()).to(device)
    xy_head = MLPXYHead(input_shape=f, normalizer=normalizer).to(device)
    subgoal = SubgoalPredictor(f).to(device)

    info = load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None,
                           device=device, strict=False)
    if "xy_head_state_dict" in info:
        xy_head.load_state_dict(info["xy_head_state_dict"])
    sg_state = torch.load(sg_ckpt, map_location=device, weights_only=False)
    subgoal.load_state_dict(sg_state["subgoal"])  # warm-start from the frozen baseline
    print(f"[cotrain] f={f} N={N} epochs={epochs} | joint fine-tune (shared latent)", flush=True)

    # STAGED unfreeze: encoder/predictor frozen (lr 0) for the first freeze_epochs
    # (refine the heads on the proven latent first), then a GENTLE encoder lr so the
    # shared latent adapts without wrecking the warm-started subgoal head.
    opt = AdamW([
        {"params": list(enc.parameters()) + list(pred.parameters()) + list(idm.parameters()), "lr": 0.0},
        {"params": list(xy_head.parameters()) + list(subgoal.parameters()), "lr": 1e-3},
    ], weight_decay=1e-5)
    print(f"[cotrain] staged: {freeze_epochs} heads-only epochs, then encoder lr={enc_lr}", flush=True)
    aux_coeff = float(m.get("aux_pos_coeff", 0.5)) or 0.5
    sg_coeff = 1.0
    nsteps = int(m.nsteps)

    for epoch in range(epochs):
        opt.param_groups[0]["lr"] = 0.0 if epoch < freeze_epochs else enc_lr
        jepa.train(); t0 = time.time()
        jl = sgl = al = 0.0; nb = 0
        for x, a, loc, _, _ in loader:
            x = x.to(device, non_blocking=True); a = a.to(device, non_blocking=True)
            loc = loc.to(device, non_blocking=True)
            B, _, T = loc.shape
            _, (jepa_loss, regl, _, _, pl) = jepa.unroll(
                x, a, nsteps=nsteps, unroll_mode="autoregressive", ctxt_window_time=1,
                compute_loss=True, return_all_steps=False)
            z = jepa.encode(x)                                   # [B,f,T,1,1] (grad)
            z_flat = z.permute(0, 2, 1, 3, 4).reshape(B * T, f)
            goal = loc[:, :, -1:].expand(B, 2, T).permute(0, 2, 1).reshape(B * T, 2)
            idx = torch.clamp(torch.arange(T, device=device) + N, max=T - 1)
            label = loc[:, :, idx].permute(0, 2, 1).reshape(B * T, 2)
            sg_loss = F.mse_loss(subgoal(z_flat, goal), label)
            aux = F.mse_loss(xy_head(z[:, :, :1]), loc[:, :, :1])
            total = jepa_loss + aux_coeff * aux + sg_coeff * sg_loss
            opt.zero_grad(); total.backward(); opt.step()
            jl += float(jepa_loss); sgl += float(sg_loss); al += float(aux); nb += 1
        print(f"[cotrain] epoch {epoch} {time.time()-t0:.0f}s jepa={jl/nb:.4f} "
              f"subgoal={sgl/nb:.5f} aux={al/nb:.5f}", flush=True)
        save_checkpoint(Path(out_dir) / "latest.pth.tar", model=jepa, epoch=epoch,
                        xy_head_state_dict=xy_head.state_dict())
        torch.save({"subgoal": subgoal.state_dict(), "N": N, "f": f},
                   Path(out_dir) / "subgoal.pth.tar")
    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[cotrain] DONE -> {out_dir}/(latest.pth.tar, subgoal.pth.tar)", flush=True)


if __name__ == "__main__":
    main()
