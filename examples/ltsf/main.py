"""LTSF — SSL pretraining entrypoint (Track 5: long-term forecasting on ETT).

Research question: does latent prediction with anti-collapse regularization learn
more transferable features than direct forecasting on noisy non-stationary series?

The DATA + TRAINING LOOP are provided. The three modelling pieces you implement
are marked `# TODO` below — that is the whole point of the track:
  1. the 1D encoder over [B, C, L] (the input window)
  2. the SSL objective (predictive JEPA  *or*  two-view VICReg)
  3. (eval.py) the downstream forecast probe + metric

Run:  python -m examples.ltsf.main --fname examples/ltsf/cfgs/train.yaml
"""
import os
import sys

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.ltsf.dataset import LTSFConfig, make_ssl_loaders

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: RNNPredictor (GRU/RNN core), Projector (MLP)
#   eb_jepa.losses:        VCLoss (variance+covariance), VICRegLoss (inv+var+cov)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    """TODO: return a 1D encoder mapping an input window [B, C=in_channels, L] to
    a representation. Expose `.represent(x) -> [B, D]` (global pooled) and, if you
    go for the predictive objective, `.frames(x) -> [B, F, D]` (a short latent
    sequence over the window) plus an `.out_dim` / `.n_frames` attribute.

    Hints: a strided Conv1d stack + global average pool is a strong baseline;
    eb_jepa.architectures has 2D encoders to take inspiration from, not a 1D one.
    For `.frames()`, split the window into F chunks along time and encode each."""
    raise NotImplementedError("TODO: build the 1D encoder (see docstring)")


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """TODO: return an nn.Module exposing `compute_loss(batch) -> (loss, logs)`.
    Pick one:
      * predictive JEPA : encode frames, roll eb_jepa RNNPredictor from a context
        frame to predict future frame latents vs an EMA target encoder; add VCLoss
        (anti-collapse) on the online latents. (batch = one input window x)
      * two-view VICReg : two augmented views -> encoder.represent -> Projector ->
        VICRegLoss. (batch = (v1, v2); adapt the dataset to return two views)
    Keep the anti-collapse term — it is the ingredient direct forecasting lacks.
    Note: SSL sees the INPUT WINDOW only (no horizon, no labels)."""
    raise NotImplementedError("TODO: assemble the SSL objective (see docstring)")


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/ltsf/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = LTSFConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader, _ = make_ssl_loaders(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        print(f"[ltsf] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[ltsf] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/ltsf/cfgs/train.yaml"
    run(fname=fname)
