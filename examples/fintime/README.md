# FinTime — JEPA on noisy financial multivariate time series (Track 5)

**Question.** Does *latent prediction with anti-collapse regularization* learn more
**transferable** features than *direct forecasting* on noisy financial series?

## Data
`thesven/fintime-decoder-dataset` — 697 instruments, 2010–2025. Each window is
`[87 variates, 64 daily steps]` (OHLCV + ~80 technical indicators + calendar,
per-feature z-scored), with ready-made targets (`direction`, `return`).
Prepared once into memmaps at
`/lustre/work/pdl17890/udl806719/datasets/Finance/fintime_prep` (see `prepare.py`).

## Layout
```
eb_jepa/datasets/fintime/   dataset.py (provided loader) + data_config.yaml
examples/fintime/
  main.py     SSL pretraining — TODO: build_encoder() + build_ssl()
  eval.py     downstream probe — TODO: probe() + metric
  prepare.py  one-time HF-parquet -> memmap (provided)
  configs/    train.yaml, eval.yaml
```

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` — a 1D encoder over `[B, C, T]` (`represent()`, and
   `frames()` if you go predictive).
2. `main.py:build_ssl` — the SSL objective: predictive JEPA (eb_jepa `RNNPredictor`
   + EMA target + `VCLoss`) **or** two-view VICReg (`Projector` + `VICRegLoss`).
3. `eval.py:probe` — the frozen-feature probe + metric, compared to a random-encoder
   floor and a supervised end-to-end baseline.

Everything else (data loading, training loop, feature extraction) is provided.

## Run
```bash
python -m examples.fintime.prepare           # once
python -m examples.fintime.main --fname examples/fintime/cfgs/train.yaml
python -m examples.fintime.eval --ckpt <.../latest.pth.tar> --target direction
```
