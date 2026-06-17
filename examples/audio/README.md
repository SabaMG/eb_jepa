# Audio — JEPA for speech keyword spotting (raw waveform vs log-mel)

**Task.** 35-keyword spotting on **Speech Commands v2** via a **frozen SSL
encoder** + linear probe. The features are learned with **no labels**; the labels
only appear in the downstream probe.

**Question.** Does the audio JEPA gain more from modeling the **raw temporal
signal** (1D waveform) or from a **time-frequency front-end** (2D log-mel)? Both
variants use the same SSL objective, the same augmentation, and the same probe
protocol, so any gap reflects the **representation + encoder**, not the augmentation.

## Data
**Speech Commands v2** (`v0.02`, ~105 k clips of 1 s @ 16 kHz, 35 keyword classes).
WAVs at `/lustre/work/pdl17890/udl806719/datasets/speech_commands_v2`; official
splits via `validation_list.txt` / `testing_list.txt`. Two input representations:
`raw -> [1, 16000]`, `mel -> log-mel [1, 64, ~101]` (per-sample standardized).

## Layout
```
eb_jepa/datasets/audio/   dataset.py (provided loader + augmentation) + data_config.yaml
examples/audio/
  main.py     SSL pretraining — TODO: build_encoder() + build_ssl()
  eval.py     downstream probe — TODO: probe() + 35-way accuracy
  configs/    train.yaml, eval.yaml
```

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` — the audio encoder over the chosen representation
   (`mode=raw`: a 1D Conv stack over `[B,1,16000]`; `mode=mel`: a small 2D mel-CNN /
   1-channel ResNet18 over `[B,1,64,T]`). Expose `represent()`, and `frames()` if
   you go predictive.
2. `main.py:build_ssl` — the SSL objective: **two-view VICReg** (eb_jepa `Projector`
   + `VICRegLoss`) **or** **predictive JEPA** (frame encoder + EMA target +
   predictor + `VCLoss` anti-collapse).
3. `eval.py:probe` — the frozen-feature linear probe + 35-way accuracy on the
   official test split, compared to a random-encoder floor and a supervised baseline.

Everything else (data loading, augmentation, training loop, feature extraction) is
provided. Reuse the eb_jepa core (`Projector`, `VICRegLoss`, `VCLoss`,
`RNNPredictor`) — do not duplicate it.

## Run
```bash
python -m examples.audio.main --fname examples/audio/cfgs/train.yaml            # raw
python -m examples.audio.main --fname examples/audio/cfgs/train.yaml data.mode=mel model.mode=mel
python -m examples.audio.eval --ckpt <.../latest.pth.tar>
```

**Metric.** Linear-probe **test accuracy** on the 35-class set (chance **2.86 %**).
