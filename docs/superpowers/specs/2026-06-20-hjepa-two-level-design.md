# Strict 2-level H-JEPA — design spec

**Date:** 2026-06-20
**Status:** implemented (local modules tested 9/9); cluster training/eval pending
**Supersedes (for the hierarchy direction):** the 1-abstraction quasimetric plan
`2026-06-20-latent-subgoal-planning.md`. That design was hierarchical *control* on a
single abstraction; this one adds a genuine **second learned abstraction** — the
LeCun H-JEPA "stack of JEPAs."

## Goal

A maze planner with **two levels of representation** (not just two time scales): a
frozen fine world model plus a learned coarse JEPA on top, both predicting in latent
space, planning by **WM imagination ("dreams")**. No A\* in the decision loop.

## The two levels

- **Level 0 (fine, FROZEN).** The existing WM: `z = encode(o)`, `predictor(z, a)` one
  cell-step. Also the wall-aware simulator the coarse targets are dreamed from.
- **Level 1 (coarse, LEARNED — the 2nd abstraction).**
  - `CoarseEncoder psi: z -> s` — compresses a fine latent to a coarse state.
  - `CoarsePredictor P_high: (s, option) -> s_hat` — predicts the coarse state after a
    **k-step macro-option** (commit to one of 4 cardinals for k fine steps).

Two abstractions (`z`, `s`), two predictors (fine frozen, coarse learned), two time
scales (1 step, k steps), both energy-based. This is strict H-JEPA, not H-JEPA-lite.

## Training the coarse JEPA — by WM dreams (no `fine_kstep_target`)

The coarse level is trained on the fine WM's **imagined latent rollouts**:

- `dream_macro_option(jepa, z0, o, k)` rolls the frozen **predictor** k steps under
  cardinal `o` **entirely in latent space** (never decoding to pixels), early-stopping
  per-sample when the latent stops moving (wall -> WM predicts "stay"). This *replaces*
  `fine_kstep_target`, which re-encoded through `jepa.unroll` and always rolled a fixed
  straight line; the dream stays in representation space and lands at walls.
- For each start `z_t` and each option `o`: target `s_target^o = psi_ema(dream(z_t,o))`
  (EMA target encoder, stop-grad — BYOL/JEPA style). Prediction `s_hat^o = P_high(s_t,o)`.
- **Loss:** `mean_o ||s_hat^o - s_target^o||^2 + std_coeff·HingeStd(s_t) + cov_coeff·Cov(s_t)`.
  The VICReg terms (the codebase's `HingeStdLoss`/`CovarianceLoss`) prevent collapse —
  the standard JEPA anti-collapse, reused not reinvented.
- EMA update `psi_ema <- tau·psi_ema + (1-tau)·psi` each step. WM frozen throughout.

## Planning — 2-level, A\*-free

```
z_goal = encode(target_obs);  s_goal = psi(z_goal)
loop:
    z_t = encode(obs);  s_t = psi(z_t)
    # HIGH (coarse, every m steps): plan over macro-options with the coarse predictor
    o*, s_sg = coarse_beam(P_high, s_t, s_goal, horizon=Hc, width=W)   # s_sg = coarse subgoal
    # LOW (fine, each step): cardinal whose 1-step fine prediction lands nearest s_sg in coarse space
    a* = argmin_a || psi(predictor(z_t, a)) - s_sg ||
    obs = env.step(a*)
```

The coarse level reasons over `Hc·k` fine cells in `Hc` cheap coarse steps (the
abstraction beats the long-horizon saturation that sank the flat planner). The fine
level grounds execution and is wall-aware, correcting optimistic coarse guesses via
closed-loop replanning every `m` steps.

## Components & files

| unit | file | tested |
|---|---|---|
| `dream_macro_option`, `CoarseEncoder`, `CoarsePredictor`, `ema_update`, `coarse_jepa_loss`, `coarse_beam`, `pick_fine_action` | `eb_jepa/hjepa.py` | `tests/test_hjepa.py` (9/9) |
| coarse JEPA trainer (dreams) | `examples/ac_video_jepa/maze/train_coarse.py` | cluster |
| 2-level A\*-free eval (seeded, SPL) | `examples/ac_video_jepa/maze/eval_hjepa.py` | cluster |
| sbatch train+eval | `rerun_hjepa.sh` | cluster |

## Risks & open questions

- **Coarse collapse / uninformative `s`** — main risk. Mitigated by VICReg; validate by
  checking `std(s) > 0` and that `||s_t - s_goal||` decreases along an A\* path. If `s`
  collapses, raise `std_coeff`.
- **Coarse distance is Euclidean in `s`-space (MVP).** Upgrade: a coarse quasimetric
  (MRN/IQE) if Euclidean ranking of options is weak.
- **Macro-option = 4 cardinals (MVP).** Upgrade: learned/VQ options (skills emerge from
  the dream) for a purer abstraction — more code, deferred.
- **k (macro horizon) and Hc, m, W** — tune on the cluster; start k=5, Hc=4, m=3, W=4.
