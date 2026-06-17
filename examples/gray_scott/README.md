# Gray-Scott — temporal JEPA on a PDE reaction-diffusion field (The Well)

**Question.** Can a JEPA learn the *dynamics* of a PDE by predicting the *latent*
of the future (not the pixels), and how does latent-space prediction compare —
in The Well's field-space VRMSE — to neural-operator surrogates (FNO / U-Net)?

## Data
`polymathic-ai/gray_scott_reaction_diffusion` (The Well,
[Ohana et al. 2024, arXiv:2412.00568](https://arxiv.org/abs/2412.00568)). Two
chemical fields **A** and **B** diffuse and react on a 128x128 grid; each
trajectory is **1001 timesteps**, stored as HDF5 under `t0_fields/{A,B}`. Feed/
kill parameters (F, k) give **6 visually distinct regimes** (spots, worms,
maze-like, ...) — one regime per HDF5 file. A training item is a clip of
`n_frames=16` with `time_stride=4`, the two fields stacked as channels into a
`[2, T, 128, 128]` tensor, z-scored per channel. The train/valid/test splits are
the dataset's own trajectory folders, so any probe is trajectory-disjoint.

## Layout
```
eb_jepa/datasets/gray_scott/   dataset.py (provided HDF5 loader) + data_config.yaml
examples/gray_scott/
  main.py     temporal-JEPA pretraining — TODO: build_encoder() + build_jepa()
  eval.py     field-space VRMSE rollout — TODO: build_decoder() + vrmse metric
  configs/    train.yaml, eval.yaml
```

## The model — temporal / predictive JEPA (not two-view)
```
context  z[:, :context_length=2]  --predictor(ResUNet)-->  z_hat (future latent)
target   z_target = target_encoder(future frames)        (EMA, no grad)
loss     = || z_hat - z_target ||  (SquareLossSeq) + VCLoss(std, cov)  (anti-collapse)
```
There is **no pixel loss in pretraining** — the model predicts a *representation*
of the future. A latent->field decoder is added only at eval to score VRMSE.

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` — a 2D frame encoder `[B, 2, H, W] -> [B, D, h, w]`
   (point at `eb_jepa.architectures.ResNet5` / `ImpalaEncoder`; stride-1 keeps the
   latent full-resolution so a decoder can map it back to a field).
2. `main.py:build_jepa` — the temporal-JEPA assembly: `eb_jepa.jepa.JEPA` with the
   shared encoder + EMA target, a `StateOnlyPredictor(ResUNet(2D, hpre, D))` that
   rolls latents forward, `VCLoss` (anti-collapse) and `SquareLossSeq` (prediction).
3. `eval.py:build_decoder` — a frozen-JEPA latent->field decoder (train it to
   minimise `MSE(decode(encode(field)), field)`); its error is JEPA's irreducible
   field floor.
4. `eval.py:vrmse_per_horizon` — multi-step **VRMSE** (variance-scaled RMSE,
   aggregated num/den) for JEPA vs **persistence** (and optionally FNO / U-Net
   surrogates, trained iso-protocol) over horizons `1..H`.

Everything else (HDF5 loading, training loop, autoregressive latent rollout
extraction) is provided. Reuse the eb_jepa core (`ResNet5`, `ResUNet`,
`StateOnlyPredictor`, `Projector`, `VCLoss`, `SquareLossSeq`, `JEPA`) — do not
duplicate it.

## Run
```bash
python -m examples.gray_scott.main --fname examples/gray_scott/cfgs/train.yaml
python -m examples.gray_scott.eval --ckpt <.../latest.pth.tar> --H 10
```
