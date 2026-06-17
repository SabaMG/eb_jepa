"""Audio — SSL pretraining entrypoint (audio/speech SSL: keyword spotting).

Research question: can a JEPA learn audio features (no labels) that linearly probe
to strong 35-keyword accuracy — and does it gain more from the raw temporal signal
(raw 1D waveform) or from a time-frequency front-end (log-mel)?

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement are
marked `# TODO` below — that is the whole point of the track:
  1. the encoder over the chosen input representation (raw waveform [B,1,16000] or
     log-mel [B,1,n_mels,T])
  2. the SSL objective (two-view VICReg  *or*  predictive JEPA)
  (eval.py adds the third TODO: the downstream linear probe + metric.)

Run:  python -m examples.audio.main --fname examples/audio/cfgs/train.yaml
      python -m examples.audio.main --fname examples/audio/cfgs/train.yaml mode=mel
"""
import os
import sys

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.audio.dataset import AudioConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP from a '256-512-128' spec), RNNPredictor
#   eb_jepa.losses:        VICRegLoss (inv+var+cov), VCLoss (variance+covariance)


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
def build_encoder(cfg):
    """TODO: return an audio encoder for the chosen input representation
    (cfg.mode == 'raw' or 'mel'). Expose `.represent(x) -> [B, D]` (global pooled)
    and an `.out_dim` attribute; if you go predictive, also `.frames(x) -> [B, F, D]`
    (a short latent sequence) plus an `.n_frames` attribute.

    Hints:
      * raw : a strided Conv1d stack over the waveform [B,1,16000] -> global avg
        pool. A wide first kernel (M5 / wav2vec-style, e.g. k=80, stride 4) then a
        few k=3 strided blocks is a strong baseline.
      * mel : a small 2D-CNN (or a 1-channel torchvision ResNet18) over the log-mel
        "image" [B,1,n_mels,T] -> global avg pool. (the mel-CNN variant.)
      eb_jepa.architectures has 2D encoders to take inspiration from, not a 1D one."""
    raise NotImplementedError("TODO: build the audio encoder (see docstring)")


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
def build_ssl(encoder, cfg):
    """TODO: return an nn.Module exposing `compute_loss(batch) -> (loss, logs)`.
    Pick one:
      * two-view VICReg : two augmented views (v1, v2) -> encoder.represent ->
        eb_jepa Projector -> VICRegLoss(std_coeff, cov_coeff). The dataset already
        returns (v1, v2, y) in ssl mode; the augmentation makes the two views.
      * predictive JEPA : cut the input into frames, roll a predictor (eb_jepa
        RNNPredictor or a small Transformer) to predict masked/future frame latents
        vs an EMA target, plus VCLoss (anti-collapse) on the online latents.
        (set data.n_views=1 so the dataset returns a single view.)
    Keep the anti-collapse term — it is what stops the features from degenerating."""
    raise NotImplementedError("TODO: assemble the SSL objective (see docstring)")


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/audio/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = AudioConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode_ssl = "ssl"
    loader = make_loader(dcfg)
    print(f"[audio] mode={dcfg.mode} | {dcfg.n_classes} keyword classes | "
          f"train(ssl)={len(loader.dataset)} | device={device}", flush=True)

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
        print(f"[audio] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[audio] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    fname = sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv \
        else "examples/audio/cfgs/train.yaml"
    overrides = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    run(fname=fname, **overrides)
