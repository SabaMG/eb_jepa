# Latent Subgoal Planning — design spec

**Date:** 2026-06-20
**Authors:** Tristan (+ Claude)
**Status:** draft for team review
**Scope:** replace the A\*-cloning high level and the constant-action low level of the
maze hierarchy with subgoals that **emerge from the frozen world model's own energy**.
No A\* anywhere in the decision loop (and none even as a *planner* at train time).

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

## 3. The one new learned component — a wall-aware latent distance `d(z_a, z_b)`

Everything hinges on a distance that respects walls. This is the only new trained piece;
the WM stays frozen (the organiser proved moving it erodes wall-awareness).

- **Interface:** `LatentDistanceHead(z_a, z_b) -> R>=0`, mirroring `GoalValueHead`'s
  pooled-latent interface (mean-pool `[B,C,T,1,1]` over spatial dims, concat, MLP).
- **Meaning:** approximate number of low-level steps to get from `z_a` to `z_b` under the
  frozen WM, *through the maze* (walls included).
- **Self-supervised label (no A\* planner):** along any stored trajectory of states
  `s_0..s_T`, encode to latents `z_0..z_T` with the frozen WM; for `i < j`, the temporal
  gap `j - i` is a sample of the step-distance between `z_i` and `z_j`. Train
  `d(z_i, z_j) ≈ j - i` (Huber regression). Because the maze trajectory data is generated
  by A\* (shortest paths), `j - i` ≈ true shortest-path distance — but **A\* is only the
  source of the training data, never a planner**, exactly the same "teacher at train"
  contract the headline already relies on. (A fully A\*-free variant trains `d` on the
  frozen WM's own random-action rollouts; noted as an option in §7.)
- **Relation to existing code:** `d ≈ -log_gamma(V)` where `V` is the organiser's
  `GoalValueHead`. We can either (a) train a fresh `LatentDistanceHead`, or (b) reuse
  their value head and use `cost = 1 - V`. We default to (a) for a clean, general pairwise
  `d`; (b) is a fast fallback.

### The make-or-break risk: long-range distance (saturation)

The flat planner gave 0% because its value **saturated** over long mazes (no gradient).
A learned `d` will have the *same* failure if it only resolves short ranges. The base WM
trains on windows of length 49, but mazes need ~120 steps.

**Mitigations (must do, not optional):**
- Train `d` on **full-length A\* trajectories** (encode the whole path, labels up to
  ~120), not the 49-clipped training windows.
- Use a **quasimetric architecture** (MRN or IQE) so `d` respects the triangle inequality
  and extrapolates monotonically, instead of a plain MLP. (MVP can start with an MLP and
  upgrade if long-range monotonicity fails validation.)
- **Validate before trusting:** on held-out mazes, `d(z_t, z_goal)` must decrease
  monotonically along the A\* path (§8). If it doesn't, the planner can't work — fix `d`
  first.

## 4. Components (isolated units)

| unit | file | what it does | depends on |
|---|---|---|---|
| `LatentDistanceHead` | `eb_jepa/state_decoder.py` (next to `GoalValueHead`) | `d(z_a, z_b) -> R>=0`; pooled-latent MLP/quasimetric | torch only |
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
- `fine_kstep_target` constant-action lookahead -> beam rollout that varies actions (§5),
  scored by learned `d` not probe-Euclidean.

**Delete from the decision loop:**
- probe-Euclidean scoring, blocked-skip, revisit penalty, no-U-turn (all obsolete once
  `d` is wall-aware).

We keep `eval_subgoal.py` and `SubgoalPredictor` in the repo **as the baseline** to A/B
against — we do not break the 66% path.

## 7. Training plan

1. **Train `d` (head only, WM frozen).** `train_distance.py`: stream maze episodes, encode
   full A\*-path states with the frozen WM, sample `(i, j)` pairs, regress `d -> j - i`
   (Huber). Add a few cross-trajectory far pairs as large-distance negatives. Fast (one
   small head, frozen encoder). Output: `distance.pth`.
   - *A\*-free option:* replace A\*-path data with frozen-WM random-action rollouts; labels
     are rollout step gaps. More honest "no A\* even in data", but sparser coverage —
     evaluate if time permits.
2. **No WM retraining.** Use the existing `aux/latest.pth.tar`.
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
- **Phase 2 — polish.** Quasimetric head if long-range monotonicity is weak; tune
  `H, m, W`; optional reuse of `GoalValueHead`.
- **Future (out of scope):** strict H-JEPA — a learned coarse encoder `psi(z)` and
  distance in coarse latent space for a true multi-abstraction hierarchy.

## 10. Open questions

1. **Quasimetric vs MLP for `d`** — start MLP, upgrade to MRN/IQE only if the §8
   monotonicity gate is weak? (Default: yes, start simple.)
2. **`z_sg` from WM rollout vs from a fresh encode** — the subgoal latent comes from the
   WM's own prediction, so it lives on the WM's predicted manifold (consistent with the
   low level, which also rolls the WM). Confirm this is stable over `m` steps.
3. **Reachability = distance budget vs uncertainty** — we operationalize the "predictive
   horizon" as `depth <= H` (a reach budget). A truer #1 uses WM-ensemble disagreement as
   the horizon. Budget is the MVP; ensemble is a stretch goal.
4. **Reuse organiser's `GoalValueHead`** as `d = 1 - V` to save training a head — worth a
   quick try in Phase 0 before committing to a fresh head?
