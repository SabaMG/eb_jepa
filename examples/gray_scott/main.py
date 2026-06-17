"""Gray-Scott — temporal-JEPA pretraining entrypoint (PDE reaction-diffusion).

Research question: can a JEPA learn the *dynamics* of a PDE by predicting the
*latent* of the future (not the pixels)? Each simulation is a 2D physical video
``[2, T, 128, 128]`` (chemical fields A, B). This is a PREDICTIVE / temporal
JEPA (video-style), NOT a two-view objective:

  context  z[:, :context_length]  --predictor-->  z_hat  (future latent)
  target   z_target = target_encoder(future frames)      (EMA, no grad)
  loss     = || z_hat - z_target ||  (latent prediction) + VC(z) (anti-collapse)

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement are
marked ``# TODO`` below — that is the whole point of the track:
  1. the 2D encoder over a frame  ``[B, 2, H, W] -> [B, D, h, w]``
  2. the temporal-JEPA assembly (encoder + EMA target + predictor + VCLoss)

Run:  python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml
"""
import os
import sys
import time

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.gray_scott.dataset import GrayScottConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: ResNet5 / ImpalaEncoder (2D encoders), ResUNet
#                          (latent->latent predictor backbone),
#                          StateOnlyPredictor (rolls latents forward), Projector
#   eb_jepa.losses:        VCLoss (variance+covariance anti-collapse), SquareLossSeq
#   eb_jepa.jepa:          JEPA (online+target encoder, predictor, .unroll(...))


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    """TODO: return a 2D frame encoder mapping a frame ``[B, 2, H, W]`` (the two
    chemical fields) to a latent ``[B, D, h, w]``. It must also accept the 5D clip
    ``[B, 2, T, H, W]`` and return ``[B, D, T, h, w]`` (the eb_jepa 2D encoders do
    this via ``TemporalBatchMixin`` — they fold T into the batch automatically).

    Hints: ``eb_jepa.architectures.ResNet5(in_d=2, h_d=henc, out_d=dstc)`` is the
    drop-in choice — stride-1 / no avg-pool keeps the latent at full ``h=w=128``
    resolution (so a decoder can later map it back to a field). ``ImpalaEncoder``
    is the heavier alternative. Expose ``out_d`` (= D = dstc) for downstream use."""
    raise NotImplementedError("TODO: build the 2D frame encoder (see docstring)")


# --------------------------------------------------------------------------- #
# 2) TEMPORAL-JEPA ASSEMBLY  — # TODO
# --------------------------------------------------------------------------- #
def build_jepa(encoder, cfg):
    """TODO: assemble and return an ``eb_jepa.jepa.JEPA`` (predictive/temporal,
    NOT two-view). The pieces, all reused from eb_jepa:
      * online + target encoder: pass ``encoder`` as both — JEPA keeps an EMA copy
        of the target internally (no-grad target of the future latent).
      * predictor that ROLLS LATENTS FORWARD: wrap a ``ResUNet(2*D, hpre, D)`` in
        ``StateOnlyPredictor(..., context_length=2)`` — it predicts the next latent
        from the previous two (state-only, no actions).
      * anti-collapse: ``VCLoss(std_coeff, cov_coeff, proj=Projector("D-4D-4D"))``.
      * prediction loss: ``SquareLossSeq(projector)`` on the projected latents.
    Build via ``JEPA(encoder, encoder, predictor, regularizer, predcost)``; the
    training loop below drives it with ``jepa.unroll(x, actions=None, ...)``.
    Keep the VC anti-collapse term — it is what stops the latent from collapsing."""
    raise NotImplementedError("TODO: assemble the temporal JEPA (see docstring)")


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/gray_scott/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = GrayScottConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    train_loader = make_loader(dcfg)
    val_loader = make_loader(GrayScottConfig(**{**dcfg.__dict__, "split": "valid",
                                                "epoch_size": dcfg.batch_size * 10}), shuffle=False)
    print(f"[gs] {len(train_loader.dataset.files)} train hdf5 | "
          f"clip=[{dcfg.channels},{dcfg.n_frames},{dcfg.img_size},{dcfg.img_size}] "
          f"stride={dcfg.time_stride} | {len(train_loader)} steps/epoch", flush=True)

    encoder = build_encoder(cfg.model).to(device)
    jepa = build_jepa(encoder, cfg.model).to(device)
    print(f"[gs] params: {sum(p.numel() for p in jepa.parameters()) / 1e6:.2f}M", flush=True)

    opt = torch.optim.Adam(jepa.parameters(), lr=cfg.optim.lr)
    use_amp = bool(cfg.training.use_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.training.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    gstep = 0
    for epoch in range(cfg.optim.epochs):
        jepa.train()
        t0 = time.time()
        for batch in train_loader:
            x = batch["video"].to(device, non_blocking=True)        # [B,2,T,H,W]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                _, (jepa_loss, regl, _, _, pl) = jepa.unroll(
                    x, actions=None, nsteps=cfg.model.steps,
                    unroll_mode="parallel", compute_loss=True, return_all_steps=False)
            if scaler.is_enabled():
                scaler.scale(jepa_loss).backward(); scaler.step(opt); scaler.update()
            else:
                jepa_loss.backward(); opt.step()
            gstep += 1
            if gstep % cfg.logging.log_every == 0:
                print(f"e{epoch} s{gstep} loss={jepa_loss.item():.4f} "
                      f"vc={regl.item():.4f} pred={pl.item():.4f}", flush=True)

        # val
        jepa.eval(); vl = 0.0; nb = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["video"].to(device)
                with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                    _, (jl, _, _, _, _) = jepa.unroll(x, actions=None, nsteps=cfg.model.steps,
                                                      unroll_mode="parallel", compute_loss=True)
                vl += jl.item(); nb += 1
        print(f"[epoch {epoch}] {time.time() - t0:.0f}s | val_loss={vl / max(nb, 1):.4f}", flush=True)
        torch.save({"epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "jepa": jepa.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[gs] done -> {ckpt_dir}/latest.pth.tar", flush=True)


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/gray_scott/cfgs/train.yaml"
    run(fname=fname)
