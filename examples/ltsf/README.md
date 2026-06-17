# LTSF — JEPA on long-term time-series forecasting (ETT) (Track 5)

**Question.** Does *anti-collapse latent prediction* learn more **transferable**,
less-overfit features than *direct forecasting* — and can it beat a simple **linear
baseline** — on noisy non-stationary signals?

PoC on the **ETT** long-term-forecasting (LTSF) benchmark, following the canonical
Time-Series-Library protocol so the MSE/MAE are comparable to DLinear / PatchTST /
iTransformer.

## Data
`ailuntz/ETT-small` — ETTh1/h2 (hourly) and ETTm1/m2 (15-min), 7 channels
(`HUFL HULL MUFL MULL LUFL LULL OT`), 2016-07 → 2018-06. Default: **ETTh1**.
Local: `/lustre/work/pdl17890/udl806719/datasets/LTSF/ETT-small/`.

Split = TSLib 12/4/4 months (train/val/test); `StandardScaler` fit on **train only**.
Borders match `thuml/Time-Series-Library` exactly. All MSE/MAE are on normalized data.

## Layout
```
eb_jepa/datasets/ltsf/   dataset.py (provided loader) + data_config.yaml
examples/ltsf/
  main.py     SSL pretraining — TODO: build_encoder() + build_ssl()
  eval.py     forecast probe  — TODO: probe() + (optional) dlinear_baseline()
  configs/    train.yaml, eval.yaml
```

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` — a 1D encoder over the input window `[B, 7, L]`
   (`represent() -> [B, D]`, and `frames() -> [B, F, D]` if you go predictive).
2. `main.py:build_ssl` — the SSL objective: predictive JEPA (eb_jepa `RNNPredictor`
   + EMA target + `VCLoss`) **or** two-view VICReg (`Projector` + `VICRegLoss`).
   SSL sees the input window only (no horizon, no labels).
3. `eval.py:probe` — the frozen-feature forecast head + metric, compared to a
   random-encoder floor and a supervised end-to-end baseline.
4. `eval.py:dlinear_baseline` (optional) — the NLinear/DLinear linear map `L->H`;
   on ETT this is the bar worth clearing.

Everything else (data loading, training loop, feature/forecast extraction) is provided.
Reuse the eb_jepa core (`eb_jepa.architectures.RNNPredictor`/`Projector`,
`eb_jepa.losses.VCLoss`/`VICRegLoss`) — do not reimplement it.

## Run
```bash
python -m examples.ltsf.main --fname examples/ltsf/cfgs/train.yaml
python -m examples.ltsf.eval --ckpt <.../latest.pth.tar> --pred_len 96
# full table: sweep --pred_len over {96, 192, 336, 720}
```

The bar worth clearing (per the track): a frozen-JEPA probe competitive with the
supervised encoder and with the DLinear baseline at horizons `{96, 192, 336, 720}`.
