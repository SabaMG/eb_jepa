# Strict 2-level H-JEPA Implementation Plan

> **For agentic workers:** implement task-by-task; local-logic tasks are TDD with `tests/test_hjepa.py`, cluster tasks run on Dalia via sbatch.

**Goal:** A maze planner with two learned abstractions (frozen fine WM + a learned coarse JEPA), planning by WM dreams. No A\* in the loop.

**Architecture:** `psi: z->s` and `P_high: (s, option)->s_hat` trained on `dream_macro_option` (latent WM imagination, wall-aware) with VICReg anti-collapse. Coarse beam picks a subgoal; fine WM executes it.

**Spec:** `docs/superpowers/specs/2026-06-20-hjepa-two-level-design.md`

---

## Task 1 — Level-1 modules + planner (LOCAL, DONE)

- [x] `dream_macro_option` — latent-space WM dream, wall early-stop. `eb_jepa/hjepa.py`
- [x] `CoarseEncoder` (psi), `CoarsePredictor` (P_high), `ema_update`
- [x] `coarse_jepa_loss` — MSE(pred, dreamed targets) + VICReg std/cov
- [x] `coarse_beam` (high level), `pick_fine_action` (low level)
- [x] `tests/test_hjepa.py` — **9/9 pass** (shapes, option encoding, EMA, loss+grad, collapse-penalty, dream roll + wall-stop, beam toward goal, fine descent)

Run locally: `uv run pytest tests/test_hjepa.py -v` (or any CPU torch interpreter).

## Task 2 — Coarse JEPA trainer (CLUSTER, written)

- [x] `examples/ac_video_jepa/maze/train_coarse.py` — frozen WM; per batch, dream all 4
      options from a mid-trajectory start, EMA-target coarse states, `coarse_jepa_loss`,
      EMA update. Saves `coarse.pth` = {psi, p_high, coarse_dim, k, f}.
- [ ] **Run on Dalia** (via Task 4) and confirm `[coarse] epoch N loss=… pred=… reg=…`:
      `pred` should fall (predictor learning) while `reg` stays controlled (no collapse).
      If `reg -> 0` AND `std(s)` is tiny, collapse — raise `std_coeff` in `coarse_jepa_loss`.

## Task 3 — 2-level A\*-free eval (CLUSTER, written)

- [x] `examples/ac_video_jepa/maze/eval_hjepa.py` — loads psi/P_high; `coarse_beam` ->
      subgoal every `m`; `pick_fine_action` each step; seeded; budget via `solve_a_star`;
      success via env `done`; SPL; writes `hjepa_eval.json`.
- [ ] **Confirm env API on first run:** `create_env(... rng=...)` forwarding and
      `info["target_obs"]` (both copied from `eval_subgoal.py`, expected to match).

## Task 4 — sbatch pipeline + run (CLUSTER)

- [x] `rerun_hjepa.sh` — train_coarse (k=5, 10 epochs) -> eval_hjepa (200 mazes, Hc=4, m=3, W=4), with the `EBJEPA_CKPTS` -> `ckpts` fallback.
- [ ] Push, then on Dalia: `git pull origin main && sbatch rerun_hjepa.sh`
- [ ] Read `maze_hjepa_<job>.out`:
      1. `[coarse] epoch … pred=… reg=…` — coarse JEPA learning, not collapsing.
      2. `[hjepa-eval] A*-FREE 2-level success=..%  SPL=..` — the headline vs the 66% / our 1-level numbers.

## Task 5 — validation & tuning (CLUSTER, after first run)

- [ ] **Collapse check:** if results are ~random, log `s.std()` and verify
      `||s_t - s_goal||` decreases along an A\* path; raise `std_coeff` if collapsed.
- [ ] **Sweep:** k ∈ {3,5,8}, Hc ∈ {2,4,6}, m ∈ {1,3,5}. Bigger k·Hc = longer coarse reach.
- [ ] **Upgrade if Euclidean coarse distance is weak:** swap to a coarse quasimetric
      (reuse `MRNDistanceHead` on `s`-space) — deferred, only if needed.

---

## Notes
- WM is **never** retrained (the organiser proved moving it erodes wall-awareness).
- `fine_kstep_target` is **not used** — replaced by `dream_macro_option` (pure latent dream).
- Baseline paths (`eval_subgoal.py`, the 1-level `eval_latent_subgoal.py`) are untouched, kept for A/B.
