"""EEG — SSL pretraining entrypoint (self-supervised representation learning).

Research question: can two-view invariance learning on unlabeled EEG learn
features that linearly separate *normal vs abnormal* recordings, generalizing
to held-out (patient-disjoint) subjects?

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement
are marked `# TODO` below — that is the whole point of the track:
  1. the 1D encoder over [B, C=19, T]
  2. the SSL objective (two-view VICReg  *or*  predictive JEPA)
The downstream probe + metric is the third `# TODO`, in eval.py.

Run:  python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml
"""
import os
import sys

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP), RNNPredictor (GRU)
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    """TODO: return a 1D encoder mapping an EEG window [B, C=n_channels, T] to a
    representation. Expose `.represent(x) -> [B, D]` (global pooled over time)
    and an `.out_dim` attribute. If you go for the predictive objective, also
    expose `.frames(x) -> [B, F, D]` (a short latent sequence) and `.n_frames`.

    Hints: a strided Conv1d stack (kernel 7, stride 2, BatchNorm + GELU) that
    downsamples time, followed by global average pooling, is a strong baseline
    for [B, 19, 2000]. eb_jepa.architectures has 2D image/video encoders to take
    inspiration from, not a 1D one — so this lives here."""
    raise NotImplementedError("TODO: build the 1D EEG encoder (see docstring)")


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """TODO: return an nn.Module exposing `compute_loss(batch) -> (loss, logs)`.
    Pick one:
      * two-view VICReg (natural choice): the dataset already returns (v1, v2);
        encoder.represent each view -> eb_jepa Projector -> VICRegLoss
        (invariance + variance + covariance). batch = (v1, v2).
      * predictive JEPA (optional): encode frames, roll an eb_jepa RNNPredictor
        from a context frame to predict future frame latents vs an EMA target;
        add VCLoss (anti-collapse) on the online latents.
    Keep the variance/covariance (anti-collapse) term — it is what stops the
    encoder from mapping every window to the same point."""
    raise NotImplementedError("TODO: assemble the SSL objective (see docstring)")


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/eeg/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

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
        print(f"[eeg] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[eeg] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/eeg/cfgs/train.yaml"
    run(fname=fname)
