# Latent Subgoal Planning — design spec

**Date:** 2026-06-20
**Authors:** Tristan (+ Claude)
**Status:** draft for team review
**Scope:** replace the A\*-cloning high level and the constant-action low level of the
maze hierarchy with subgoals that **emerge from the frozen world model's own energy**.
No A\* anywhere in the decision loop (and none even as a *planner* at train time).

**One-line framing:** this is **Quasimetric RL (QRL)** instantiated on a *frozen JEPA
world model* — learn a quasimetric cost-to-go in the WM's latent space, then derive
both the subgoal (high level) and the action (low level) by minimizing it. The WM is
never retrained; all learning is one small quasimetric head.

---

## 1. Motivation — what's wrong with the current hierarchy

The current A\*-free stack (organiser's `eval_subgoal.py` + `hierarchical.py`) works
(~66% / SPL 0.62) but rests on three weak ideas:

1. **The high level clones A\*.** `SubgoalPredictor(z, goal_xy)` is a supervised MLP whose
   label is *the A\* position N=4 cells ahead*. So A\* is still the brain — it's just
   moved to training time. This is the **least-JEPA** part of the system: it's behavior
   cloning of a symbolic planner, not planning in representation space.

2. **The low level rolls constant actions.** `fine_kstep_target` rolls the frozen WM `K`
   steps with **one fixed cardinal** (`LLLL` / `RRRR`). It can only see *straight*
   corridors — it cannot see a route that turns within the horizon. The beam variant
   (`fine_beam_dist`, our addition) patches this, but it's bolted onto the same scoring.

3. **Scoring is as-the-crow-flies.** Both reachers score a direction by the **probe-decoded
   Euclidean distance** from the K-step endpoint to the waypoint `xy`. The organiser's own
   `README_value.md` flags this: a geometric proxy "ignores walls between agent and goal."

We want to fix all three at once, and end up with something **more JEPA**, not less.

## 2. The core idea — subgoals as energy minimization on the frozen WM

Stop *teaching* the subgoal; *infer* it. A subgoal becomes a property of the learned
dynamics, chosen by minimizing predicted energy. Two halves (the user's ideas #1 + #2,
fused):

- **#2 alignment:** the subgoal should advance toward the goal — minimize a learned
  latent distance `d(z_sg, z_goal)`.
- **#1 reachability / predictive horizon:** the subgoal must stay inside the world
  model's reliable reach — only commit to a `z_sg` the WM can actually roll to within a
  horizon `H`.

> **The subgoal is the most goal-advancing latent still inside the world model's
> reliable predictive horizon.**

Formally, at a high-level replan:

```
z_sg* = argmin_{z_cand}  d(z_cand, z_goal)
        over candidates z_cand reachable from z_t within H steps under the frozen WM
        (optionally with depth(z_cand) >= d_min, to keep the hop multi-step).
```

Then a low-level reacher drives toward `z_sg*` by 1-step energy descent, and the high
level re-plans every `m` steps (or when `z_sg*` is reached). This is closed-loop
receding-horizon control with a learned terminal cost.

### Is it still JEPA? Is it still *hierarchical*?

- **JEPA:** yes — everything lives in the frozen WM's latent space. Candidates are WM
  rollouts (predicted latents), selection is energy minimization, the low level minimizes
  predicted-latent energy. No pixels, no symbolic planner in the loop.
- **Hierarchical (control sense):** yes — two time scales. The high level picks a latent
  target at horizon `H` (coarse, every `m` steps); the low level reaches it per step
  (fine). The high level emits a *latent* subgoal that conditions the low level. This is
  LeCun's H-JEPA *control* pattern.
- **Honest caveat:** this is *hierarchical planning over a single JEPA*, not a stack of
  JEPAs with their own learned abstractions per level. Both levels share the frozen WM's
  latent. The strict-H-JEPA upgrade (a coarse encoder `psi(z)` and distance in coarse
  space) is listed as future work (§9), not in scope here.
- **The hierarchy is a knob, not automatic:** if `m=1` and `H=1` it collapses to flat
  greedy planning. We keep `H >> 1` and `m > 1` so it stays genuinely two-level.

## 3. The one new learned component — a wall-aware latent quasimetric `d(z_a, z_b)`

Everything hinges on a distance that respects walls. This is the only new trained piece;
the WM stays frozen (the organiser proved moving it erodes wall-awareness).

### 3.1 Architecture — **IQE quasimetric** (default), not a plain MLP

A plain concat-MLP has **no metric guarantees**: nothing enforces the triangle
inequality `d(a,c) ≤ d(a,b) + d(b,c)`, so it wobbles and — critically — **extrapolates
badly to long range** (our make-or-break risk, §3.3). We therefore use a quasimetric
architecture by default:

- **Default: IQE (Interval Quasimetric Embedding)** — embed each latent into intervals,
  `d(z_a, z_b) = sum of interval lengths`. SOTA for quasimetric goal-reaching
  (Wang & Isola, 2022); guarantees the triangle inequality and is highly expressive.
- **Fallback: MRN (Metric Residual Network)** — `d = symmetric_MLP + max_k(asymmetric_k)`;
  simpler (~20 lines), same triangle-inequality guarantee. Use if IQE is fiddly.
- **Debug baseline only: plain MLP** — concat pooled latents -> MLP -> softplus. Keep for
  a 1-hour smoke test, but it is *not* the shipped head.

All three share the same pooled-latent interface as `GoalValueHead`
(`z` is `[B,C,T,1,1]`; mean-pool spatial dims -> `[B,C]`). Output is `d >= 0`.

**Why this matters:** IQE/MRN is the single architecture change that de-risks the whole
design — it turns the fragile long-range behaviour into the head's *built-in* property,
and it gives the project its SOTA framing (QRL). See §10 for alternatives we rejected.

### 3.2 Meaning and self-supervised loss

- **Meaning:** approximate number of low-level steps to get from `z_a` to `z_b` under the
  frozen WM, *through the maze* (walls included) — a learned cost-to-go.
- **Loss — temporal-gap regression.** Encode a trajectory `z_0..z_T` with the frozen WM.
  For a sampled pair `i < j`, the steps between them along the path is exactly `j - i`:

  ```
  L_reg = E_{(i,j), i<j} [ Huber( d(z_i, z_j) - (j - i) ) ]
  ```

  Huber (not MSE) so rare long pairs don't dominate. Optionally symmetrize
  (`d(z_j, z_i)` also -> `j - i`, maze is undirected) and add a few cross-trajectory pairs
  as large-distance negatives so distances don't collapse globally.
- **Why the labels are *exact*, not noisy:** a sub-path of a shortest (A\*) path is itself
  shortest between its endpoints, so within one A\* trajectory `j - i` **is** the true
  shortest-path step-distance. **A\* is only the label oracle in the data, never a planner**
  — the same "teacher at train, none at eval" contract the headline already relies on.
  (A fully A\*-free variant uses the frozen WM's own random-action rollouts and a
  contrastive/temporal loss instead of regression; noted in §7 and §10.)
- **Relation to existing code:** `d ≈ -log_gamma(V)` where `V` is the organiser's
  `GoalValueHead`. Phase-0 shortcut: reuse their head, `cost = 1 - V`, zero new training,
  just to validate the loop in ~1h before training a fresh IQE head.

### 3.3 The make-or-break risk: long-range distance (saturation)

The flat planner gave 0% because its value **saturated** over long mazes (no gradient).
A learned `d` dies the same way if it only resolves short ranges. The base WM trains on
windows of length 49, but mazes need ~120 steps.

**Mitigations (must do, not optional):**
- **IQE/MRN head** (§3.1) — triangle inequality makes the metric extrapolate monotonically
  to far pairs never seen together; this is the main defence.
- **Train on full-length A\* trajectories** — set the loader `sample_length` to the full
  path (`n_steps`) so sampled pairs span the whole 0–120 range, not the 49-clipped window.
- **Validate before trusting:** on held-out mazes, `d(z_t, z_goal)` must decrease
  monotonically along the A\* path (§8). **This is a hard gate** — if it fails, stop and
  fix `d` before building the planner.

## 4. Components (isolated units)

| unit | file | what it does | depends on |
|---|---|---|---|
| `LatentDistanceHead` | `eb_jepa/state_decoder.py` (next to `GoalValueHead`) | `d(z_a, z_b) -> R>=0`; pooled-latent **IQE** quasimetric (MRN fallback, MLP debug-only) | torch only |
| `train_distance.py` | `examples/ac_video_jepa/maze/` | self-supervised training of `d` on the **frozen** WM | frozen WM ckpt, maze data |
| `latent_subgoal_step()` | `eb_jepa/hierarchical.py` | given frozen WM + `d` + `z_t` + `z_goal` + state, return next action and current `z_sg` | WM, `d` |
| `eval_latent_subgoal.py` | `examples/ac_video_jepa/maze/` | A\*-free closed-loop eval harness (budget, SPL, GIFs) calling the planner | env, WM, `d` |

Each is independently testable: `d` by correlation with A\* distance; the planner on a
fixed maze; the eval reuses the existing harness.

## 5. Data flow (one episode)

```
reset env -> obs, info (info["target_obs"] = goal image)
z_goal = WM.encode(normalize(info["target_obs"]))          # goal latent, once
loop until done or budget exhausted:
    z_t = WM.encode(normalize(obs))
    # --- HIGH LEVEL (every m steps, or when z_sg reached) ---
    candidates = beam_rollout(WM, z_t, horizon=H, width=W)  # predicted latents + first action
    z_sg = argmin_{c in candidates, depth(c) >= d_min}  d(c, z_goal)
    # --- LOW LEVEL (every step) ---
    a* = argmin_{a in 4 cardinals}  d( WM.predictor(z_t, a), z_sg )   # 1-step energy descent
    obs, done = env.step(a*)
```

Wall handling is automatic: a blocked action makes the WM predict "stay", so its
predicted latent does not approach `z_sg` (high `d`) and is not chosen. No blocked-skip
hack, no revisit penalty, no no-U-turn rule — those were band-aids for the geometric
score and are removed.

## 6. What we keep, replace, and delete

**Keep (frozen / reused):**
- The fine WM (`aux/latest.pth.tar`) — `encode`, `predictor`, `unroll`. Never retrained.
- The env, budget (`4·A*+10`), SPL metric, GIF dump — reuse the eval harness verbatim.
- The position probe `xy_head` — **only** for the goal-cell success check and GIF labels,
  never for planning decisions.

**Replace:**
- `SubgoalPredictor` (A\* cloning) -> energy-minimization subgoal selection (§2).
- `fine_kstep_target` constant-action (`LLLL`/`RRRR`) lookahead -> beam rollout that varies
  actions (§5), scored by learned `d` not probe-Euclidean. (We keep the **beam search** —
  with discrete 4-cardinal actions an exact tree beats sampling planners like MPPI; `d`
  just turns it into a proper heuristic search. See §10 for why not MPPI / amortized policy.)

**Delete from the decision loop:**
- probe-Euclidean scoring, blocked-skip, revisit penalty, no-U-turn (all obsolete once
  `d` is wall-aware).

We keep `eval_subgoal.py` and `SubgoalPredictor` in the repo **as the baseline** to A/B
against — we do not break the 66% path.

## 7. Training plan

1. **Train `d` (IQE head only, WM frozen).** `train_distance.py`:
   ```python
   WM = build_fine(...); load aux/latest.pth.tar; WM.eval(); requires_grad_(False)
   d_head = IQEDistanceHead(dim=512); opt = AdamW(d_head.parameters(), 1e-3)
   for batch in maze_loader:                  # states:[B,C,T,H,W]; full A* paths
       with torch.no_grad():
           z = WM.encode(states)              # [B,512,T,1,1]  (frozen, no grad)
       i, j = sample_pairs(T, per_traj=K)     # i<j, vectorized
       loss = huber(d_head(z[...,i], z[...,j]) - (j - i).float())
       loss.backward(); opt.step(); opt.zero_grad()
   ```
   Cheap: encoder frozen under `no_grad`, only a small head trains (minutes). Output:
   `distance.pth`. **Loader change:** `sample_length = n_steps` (full path), so pairs span
   the full 0–120 range (§3.3).
   - *A\*-free-data option:* replace A\*-path data with frozen-WM random-action rollouts +
     a contrastive/temporal loss (no exact integer labels). More honest "no A\* even in
     data", sparser coverage — evaluate if time permits (§10).
2. **No WM retraining.** Use the existing `aux/latest.pth.tar`. The WM is off-limits.
3. **Eval** via `eval_latent_subgoal.py` (seeded, 200 mazes — see §8).

## 8. Evaluation & ablations

Reuse the existing harness but **seeded and at 200 mazes** (the 32-maze unseeded eval has
a ±13% CI — two runs of the *same* model already scored 81% and 90%). Add a `seed` arg
plumbed into `MazeEnv(rng=np.random.default_rng(seed))`.

- **Headline:** A\*-free success% / SPL vs the 66% baseline and vs our beam reacher.
- **Ablation A (the key one):** learned `d` vs probe-Euclidean scoring — isolates the
  value of Component 3.
- **Ablation B:** horizon `H` sweep, replan period `m` sweep, beam width `W`.
- **Distance validation (gate):** fraction of held-out mazes where `d(z_t, z_goal)`
  is monotonically decreasing along the A\* path; correlation of `d` with true A\* distance.
  **If this gate fails, stop and fix `d` before touching the planner.**
- **Sanity floor:** random-walk baseline (already exists, `eval_random.py`).

## 9. Phasing (hackathon-shippable)

- **Phase 0 — distance head + validation.** Train `d`, run the §8 distance gate. This
  alone is a result ("a wall-aware learned latent metric"). If the gate fails, we learned
  something and pivot.
- **Phase 1 — MVP planner.** Committed-subgoal version: high level beam-to-`H` + argmin
  `d`; low level greedy 1-step descent; re-plan every `m`. Genuinely two-level. Ship and
  measure.
- **Phase 2 — polish.** Tune `H, m, W`; swap IQE->MRN if IQE is fiddly; optional
  amortized low-level policy `pi(a|z_t,z_sg)` distilled from the search for speed.
- **Future (out of scope):** strict H-JEPA — a learned coarse encoder `psi(z)` and
  distance in coarse latent space for a true multi-abstraction hierarchy.

## 10. Rejected alternatives (and why) + open questions

**Architecture choices we deliberately did *not* take** (so we don't relitigate them):

| Component | Considered | Decision |
|---|---|---|
| Distance head | plain MLP | **Rejected as the shipped head.** No triangle inequality -> bad long-range extrapolation = our top risk. MLP kept only as a 1-hour debug baseline; **IQE** ships (MRN fallback). |
| World model | retrain bigger / different encoder | **Rejected.** Frozen and precious — co-training already proved moving it erodes wall-awareness. All architecture freedom is in the new head + planner. |
| Low-level search | MPPI / CEM (sampling planners) | **Rejected.** Actions are 4 *discrete* cardinals; an exact beam/tree dominates samplers (MPPI shines with continuous actions). Keep beam, now scored by `d`. |
| Low-level control | amortized policy `pi(a|z_t,z_sg)` | **Deferred.** Faster at inference but needs extra training; search with `d` needs none. Phase-2 speed upgrade only. |
| `d` loss | contrastive / InfoNCE / C-learning | **Deferred.** Regression on A\*-subpath gaps gives *exact* integer labels; contrastive is the path for fully A\*-free *data* — only if we drop A\*-generated trajectories. |
| Reachability test | WM-ensemble uncertainty as the horizon | **Deferred.** We operationalize "predictive horizon" as a reach budget `depth <= H` (no ensemble needed). Ensemble-uncertainty horizon is a stretch goal. |
| Goal representation | `xy` + probe | **Rejected.** Use the latent `z_goal = E(target_obs)` — more JEPA, and `d` needs a latent anyway. |

**Open questions for the team:**
1. **IQE vs MRN** — default IQE (more expressive); fall back to MRN only if IQE training
   is unstable. OK?
2. **`z_sg` from WM rollout vs fresh encode** — the subgoal latent comes from the WM's own
   prediction, so it lives on the WM's predicted manifold (consistent with the low level,
   which also rolls the WM). Confirm it's stable when held for `m` steps.
3. **Phase-0 shortcut** — try `d = 1 - V` with the organiser's *existing* `GoalValueHead`
   first (zero training, ~1h) to validate the loop before training a fresh IQE head?
