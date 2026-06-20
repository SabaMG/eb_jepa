# Hackathon Report — A\*-free maze navigation with a 2-level H-JEPA

**Team:** vivatech-equipe7 (Tristan, Lu) · **Date:** 2026-06-20
**Repo:** `SabaMG/eb_jepa` (our fork of the **official hackathon repo**, itself a fork of `facebookresearch/eb_jepa`)
**Cluster:** Dalia (IDRIS), GB200 GPUs · all runs logged to W&B (`tristan-faure-epita/eb_jepa`, group `hjepa`)

> This document is the working log + reflection for the presentation. It records what
> was given, what we found wrong with it, what we designed, the problems we hit, and
> the conceptual insights — honestly, including what does *not* yet work.

---

## 0. Executive summary

- **The given baseline (organiser's) didn't actually run end-to-end** in the streaming
  config — we found and fixed **4 real bugs** before any A\*-free number could be reproduced.
- **The "~66 %" baseline is the *organiser's* number** (in their `README_hierarchical.md`),
  **not an external benchmark and not ours.** The eval is **unseeded over only 32 mazes**
  (±13 % CI): the *same* model scored **81 % then 90 %** on two seeds — so single-number
  comparisons are noise-dominated. (What was *ours* here: the bug fixes + orchestration that
  made it run, not the algorithm — see §3.)
- We critiqued the given design and identified its **least-JEPA parts** (A\*-cloned
  subgoals, constant-cardinal lookahead, as-the-crow-flies scoring).
- We designed and built a **strict 2-level Hierarchical JEPA** — two learned abstractions,
  a coarse world model trained by **WM "dreams,"** replacing A\* with a *learned* router.
- **Status: the 2-level H-JEPA works — 0 % → 45 % success / SPL 0.23** on **200 seeded
  mazes** (the 55 %/20-maze figure was small-sample noise; 45 % is the trustworthy number).
  The path there: trains without collapse → fixed coarse-distance **saturation** (full-length
  trajectories + explicit metric) → the **GIF revealed a *frozen agent*** (memoryless greedy
  low level deadlocked on walls) → **blocked-skip**, then a **model-based, dreamer-scale wall
  avoidance** (detect walls from the WM's own "stay" dream) unfroze it. The honest framing:
  *a genuinely 2-level H-JEPA whose learned coarse router replaces A\*; it navigates ~half the
  mazes A\*-free, but **inefficiently** (SPL 0.23 vs the A\*-routed baseline's 0.62) — success
  reliability + path efficiency are the open items.*

All on the **full 21×21 maze** (img 63×63) — same size as the baseline, apples-to-apples.

---

## 1. What we were given (the organiser's code)

**Attribution (important for the talk).** We **forked the official hackathon repo** (itself a
fork of `facebookresearch/eb_jepa`). **Everything in the baseline — the maze env, the fine
WM, `hierarchical.py`, `SubgoalPredictor`, the greedy reacher, the training configs — is
organiser code** (git authors: Trick5t3r, Amir Bar, Basile Terver, Koustuv Sinha, …).
**Only changes on our fork (`SabaMG`) count as ours** — i.e. Tristan/the team's commits since
19 June ~19:30. Trick5t3r and the others are **not** on our team.

Upstream `facebookresearch/eb_jepa` is a **flat** JEPA world model + MPPI planner on a
`two_rooms` env. **No maze, no hierarchy.** The official hackathon repo added, on top:

| Component | What it is |
|---|---|
| Maze env + data | online DFS maze + A\* solver; 2-channel obs `[agent dot, wall mask]` |
| **Fine world model** | Impala encoder (z∈R⁵¹²) + RNN(GRU) predictor + VICReg regularizer + **position probe** (aux-pos), trained with `wall_bump` so it's **wall-aware** |
| `SubgoalPredictor` | MLP: `(z, goal_xy) → next waypoint xy`, **supervised on A\* waypoints** |
| Low-level reacher | `fine_kstep_target`: roll WM **K constant-cardinal steps**, score endpoint by **probe position** distance to the waypoint |
| Learned value | `GoalValueHead` V(z, z_goal) (TD-MPC), used in the *flat* MPPI planner |
| Co-training | `main_cotrain.py` (jointly fine-tune both levels) |

**Claimed result:** ~66 % success / SPL 0.62, A\*-free, 21×21, 32 mazes.
**Their own negative result:** co-training *hurts* — moving the encoder erodes the fragile
wall-aware WM. **Lesson they drew: freeze the WM, invest in the planner.** (We kept this.)

---

## 2. Problems we found *in the given code*

The hierarchical pipeline **did not run end-to-end** in the streaming-data config. Four bugs:

1. **`main_subgoal.py` never warmed up the data pipeline** → `manager.current = None` →
   `TypeError: 'NoneType' object is not subscriptable`. The job died at step 3 (subgoal),
   so **no eval ever ran** — the `eval/` and `sg/` dirs were empty.
2. **bf16/fp32 mismatch:** the stream pipeline yields bf16, the frozen encoder is fp32 →
   `RuntimeError: Input type (BFloat16) and bias type (float) should be the same`. (Base/aux
   training survived via AMP autocast; the subgoal loop had none.)
3. **`eval_subgoal.py` called `init_data` without `device`** → `ValueError` under
   `pipeline.mode='stream'`.
4. **Checkpoint path mismatch:** `env.sh` defaults `EBJEPA_CKPTS=$WORK/checkpoints` but the
   baseline was trained into `$WORK/ckpts` → "aux not found."

**Implication for the presentation:** the "proven 66 %" was *not reproducible as shipped*
in this config. Our fixes are what made the baseline runnable again — a real contribution
before any new architecture.

---

## 3. Reproducing the baseline — and the variance trap

After the fixes, we reproduced the A\*-free hierarchical eval:

| Run | mazes | success | SPL |
|---|---|---|---|
| 74856 (seed A) | 32 | **81.25 %** | 0.812 |
| 74883 (seed B) | 32 | **90.62 %** | 0.903 |

Same model, same command — a **9-point swing from 3 mazes**. Root cause: **the eval is
unseeded** (`np.random.default_rng()` with no seed; `seed:1` in the config is training-only)
and uses only **32 mazes** (≈ ±13 % 95 % CI).

**Key reframe:** the "66 %" is the **organiser's** earlier run (written into
`README_hierarchical.md`), **not an external benchmark**. So our 81–90 % vs their 66 % is a
**re-run of *their* algorithm across training seeds + a noisy eval** — nothing was "wrong."
**Methodology fix we adopted:** seed the env + evaluate on **200 mazes** for every number.

**What was actually *ours* in this reproduction.** The algorithm (greedy K-step reacher +
A\*-cloned `SubgoalPredictor`) and the configs are **organiser code, unchanged**. Our
contribution that produced the number was: **(a) the 4 bug fixes** (without them the pipeline
crashes — see §2), and **(b) the sbatch orchestration** (`recreate_baseline_full.sh`,
`rerun_sg_eval.sh`). So the 81–90 % is *their algorithm, made runnable + reproducible by us*,
**not an algorithmic improvement of ours.** (Our one real algorithmic addition at this stage,
the **beam-search reacher** `fine_beam_dist`, lives in a *separate* eval and was **not** in
the 81–90 % numbers — TODO: run beam-vs-greedy on 200 seeded mazes to quantify it.)

> ⚠️ **To confirm before the talk:** Tristan/Lu, was any *algorithmic* tweak folded into the
> 81–90 % reproduction beyond the bug fixes + scripts? Git shows the organiser algorithm
> unchanged, but flag anything local/uncommitted so we attribute it correctly.

---

## 4. What's weak in the given design (our critique)

Three "least-JEPA" / fragile choices, in the system's own terms:

1. **The high level clones A\*.** `SubgoalPredictor` is supervised regression onto A\*
   waypoints — *behavior cloning of a symbolic planner*. A\* is still the brain; it's just
   moved to training time. This is the **least-JEPA** component.
2. **The low level rolls a constant cardinal** (`LLLL`). `fine_kstep_target` can only see
   *straight* corridors — it cannot turn within the horizon.
3. **Scoring is as-the-crow-flies.** Distance = probe-decoded **position** to the waypoint —
   **ignores walls** between endpoint and waypoint (the organiser's own `README_value`
   flags this as why a learned cost was needed).

These motivated a redesign where **the world model itself does the planning**, in latent
space, with no A\* in the loop.

---

## 5. Conceptual journey — from "H-JEPA-lite" to strict 2-level H-JEPA

- **First design (1-level quasimetric).** Learn a wall-aware latent distance
  `d(z, z_goal)`; *infer* subgoals by energy minimization (no A\* cloning). Tested locally,
  spec written. **But:** this is **2 control levels on 1 abstraction** — hierarchical
  *control*, not a *stack of JEPAs*. By the strict definition it is **H-JEPA-lite**.
- **The pivot.** LeCun's H-JEPA = a **stack of JEPAs, each with its own learned abstraction
  at a coarser time scale**. So we committed to a **second learned abstraction**.
- **Design principle the user insisted on:** *use the dream with the WM* — plan by
  **imagining latent rollouts**, never decoding to pixels. That is the core of JEPA.

### Counting "levels" (the distinction that matters)
- **Levels of control / time scale:** both designs have 2 (subgoal vs action).
- **Levels of representation / abstraction:** baseline = 1; ours = **2** (`z` and `s=ψ(z)`).
  *That* is what makes ours the genuine H-JEPA.

---

## 6. Our architecture — strict 2-level H-JEPA

**Level 0 (fine, FROZEN):** the existing WM. `z = encode(o)`, `predictor(z, a)` one cell-step.
Also the wall-aware **simulator** for the coarse targets. (We keep it frozen — the
organiser proved moving it erodes wall-awareness.)

**Level 1 (coarse, LEARNED — the 2nd abstraction):**
- `ψ: z → s` (coarse encoder, R⁵¹² → R³²)
- `P_high: (s, macro-option o) → ŝ` (coarse predictor; option = commit to a cardinal for k fine steps)

**Trained by WM dreams** (`dream_macro_option`): roll the frozen **predictor k steps in
latent space** under cardinal o, **early-stopping at walls** (the WM predicts "stay"). This
**replaces `fine_kstep_target`** (which re-encoded via `unroll` and rolled a fixed straight
line). Target = `ψ_ema(dream)`, prediction = `P_high(ψ(z), o)`, with **VICReg** (std +
covariance, reused from the codebase) preventing collapse, and an **EMA target encoder**.

**Planning (A\*-free, 2-level):**
- **HIGH** (`coarse_beam`, every m steps): imagine macro-options with `P_high`, score by
  distance to `s_goal`, return the next **coarse subgoal `s_sg`** (an imagined coarse state
  ~k cells ahead toward the goal). *This replaces A\*.*
- **LOW** (`pick_fine_action`, every step): pick the cardinal whose 1-step fine prediction
  lands nearest `s_sg` in coarse space. Walls handled intrinsically (blocked → "stay" → far).

### Subgoals: ours vs theirs
| | Organiser | Ours |
|---|---|---|
| Subgoal is | an (x,y) **position** | a **coarse latent** `s_sg` |
| Comes from | `SubgoalPredictor` **cloning A\*** | the coarse WM **imagining** a macro-step (no A\*) |
| Followed by | probe **position** distance | **coarse-latent** distance |

### Our k-step dream vs their K-step lookahead
The rollout op is *similar* (roll WM k steps in latent). The **role** differs:
- **Theirs:** the K-step roll **is the runtime decision rule**; **A\* supplies the route**.
- **Ours:** the k-step dream is a **training teacher** that distills into a **learned coarse
  world model**; **the coarse model supplies the route** (replacing A\*). At eval we don't
  roll the WM k steps — `P_high` reproduces it in one cheap call and the beam plans.

> One line: *they roll the WM to **follow** an A\*-drawn route; we roll it to **train a coarse
> model that draws the route itself**.*

---

## 7. Problems we ran into with our architecture (the 0 % saga)

**Run 1 (75514):** Full pipeline runs on GPU, logs to W&B, **coarse JEPA trains without
collapse** (`s_std ≈ 2.0`, VICReg working) — but **0 % success**.

**Diagnostic (we built `diag_hjepa.py`):** walk the A\* path, measure whether the coarse
distance is navigable:
- `‖ψ(z_t) − s_goal‖` decreases toward goal only **0.68** of the time (random = 0.5).
- `coarse_beam` picks the A\*-correct direction only **0.37** (random = 0.25).
- 0.37 correct-per-step **compounds to 0 %** over a ~96-step maze.

**Diagnosis:** the coarse space was trained to **predict + not collapse**, but **nothing
made Euclidean distance mean "closer to goal."** The metric is the problem.

**Fix A (75719) — distance-shaping ranking loss + bigger k.** **No improvement** (0.65,
0.37, still 0 %). The ranking loss barely optimized.

**Root cause (the real one):** the coarse trainer inherited **`sample_length = 17`**, so the
distance was only ever trained on **≤16-step** relationships — but **goals are ~96 steps
away**. The coarse distance **saturates** for far goals. **This is the exact failure that
gave the organiser's *flat* planner 0 %** ("the value saturates over a 50-cell maze") — we
**reproduced it one level up.** Plus: 128-d Euclidean was noise-dominated.

**Fix B (76270, current):**
- **Full-length trajectories** (`sample_length = n_steps`, ~90) → distance covers the real range.
- **Direct distance regression** `‖ψ(z_i) − ψ(z_j)‖ → |i−j|` over the full range.
- **`coarse_dim` 128 → 32** for a cleaner Euclidean metric.

Early signal: the distance loss **fell** (18.6 → 5.3 vs the ranking that barely moved),
diagnostic monotonicity rose **0.68 → 0.76**. But the coarse-beam direction match **dropped
to 0.32** and `s_std` ballooned to ~4.6 — the distance term and the predictive term fight
over ψ. Eval still **0 %**.

### The real blocker, found by VISUALIZING — and the breakthrough

We added GIF dumps (like the baseline's `eval_subgoal`) and ran eval on the trained
`coarse.pth`. The GIFs showed the **agent never moves — it sits on the same cell for the
entire 219–459-step budget.** Root cause: the low level (`pick_fine_action`) was a
**memoryless greedy argmin**. The fine WM predicts "stay" into a wall, so if the best-scoring
direction is a wall and "staying" is "closest" to the subgoal, the agent **re-picks the same
blocked move every step → deadlock.** The 0 % was a *frozen* agent, not a wandering one — it
never even got to test whether the coarse routing works.

**Fix (no retrain):** `rank_fine_actions` (rank all 4 cardinals by coarse distance) +
**execution-feedback blocked-skip** (try best-first; if the agent didn't move, blacklist that
move at this cell and try the next; no immediate U-turn) — the same mechanism the organiser's
*working* reacher uses, but driven by **coarse-latent** distance instead of probe position.

**Result: 0 % → 55 % success** (11/20 mazes, SPL 0.279, same `coarse.pth`, just unfrozen).
The 2-level H-JEPA **works** — the learned coarse router reaches the goal a majority of the
time. SPL 0.279 is low (inefficient, lots of backtracking) → next: tighten paths (better
metric, larger beam) and run the seeded **200-maze** number for a solid figure.

> **Discovered purely from the GIF — visualization > guessing.** Lesson for the talk.

### Is it still H-JEPA with this fix? Yes.
The *decision* (which direction) stays fully latent: cardinals ranked by
`‖ψ(predictor(z,a)) − s_sg‖` over imagined 1-step WM predictions, toward a coarse-JEPA
subgoal — two learned abstractions, energy in latent space, no pixels, no A\*. The
blocked-skip is a **reactive control wrapper** ("did I move?"), not a planner, and it is
explicitly fair within the *A\*-free* protocol (the baseline uses the same rule). **More-JEPA
variant (noted for the slides):** detect blocked moves from the **WM's own prediction**
(`‖predictor(z,a) − z‖ ≈ 0`) *in imagination*, so wall-avoidance comes from the model and the
agent never bumps — the purest form (~5 lines).

---

## 8. Conceptual insights (slide-ready)

1. **JEPA is about *where*, not *how far*.** Prediction in **latent space, never pixels**.
   Looking many steps ahead is fine — and is the **intended use** of a JEPA world model
   (Mode-2 planning) — *as long as the lookahead stays latent*. Pixel lookahead (Dreamer-style
   reconstruction) is the thing JEPA avoids.
2. **Hierarchy = look far ahead *efficiently*.** One model unrolling 96 noisy steps is
   fragile; the coarse level looks far in a few big steps, the fine level looks ~1 step.
3. **The recurring enemy is distance/value SATURATION.** The latent is wall-dominated, so a
   single global goal-distance goes flat over long mazes. It killed the flat planner (0 %),
   and we hit it again at the coarse level. **The fix is always: train the distance over the
   full range, with a real metric** (regression / quasimetric), not a hand geometric proxy.
4. **The hard part is replacing A\*.** A\* is optimal global planning. The baseline *leans on
   it* (66 %); we try to *learn* it. More interesting, genuinely harder — and the honest
   reason ours isn't at 66 % yet.

---

## 9. Status & next steps

**Works:** the 2-level H-JEPA trains end-to-end on GPU, does not collapse, the dream +
coarse predictor + beam + fine descent all run; full W&B logging; seeded 200-maze eval.

**Open (the hard part):** make the coarse distance a reliable long-range navigation signal.
- ✅/⏳ Full-length trajectories + distance regression + small coarse_dim (run 76270).
- If still weak: **coarse quasimetric head** (MRN/IQE) instead of Euclidean (guarantees
  triangle inequality → extrapolates to far goals).
- Tune `k` (macro horizon), `Hc` (coarse beam depth), `m` (replan period).
- Learned/VQ macro-options (skills emerge from the dream) instead of 4 fixed cardinals.

**Honest bottom line for the talk:** we (a) made the given baseline *actually runnable*,
(b) exposed that its "66 %" is a noisy self-comparison, (c) critiqued its least-JEPA parts,
and (d) built the **more principled, genuinely 2-level H-JEPA** that replaces A\* with a
learned coarse world model. The architecture is sound and trains; closing the
learned-router gap to A\* is the remaining (and most interesting) problem.

## 10. Upgrade experiments & the final verdict (200 seeded mazes, 21×21)

| config | success | SPL | note |
|---|---|---|---|
| Upgrade1: Euclidean dist + macro-beam | 45.0 % | 0.230 | first working number |
| + fine dream-lookahead (depth 4) | 40.5 % | 0.212 | **hurts** (deeper search on a noisy cost) |
| Upgrade3+4: quasimetric (MRN) + LayerNorm + macro-beam | 38.0 % | 0.181 | metric ↑↑ (monotonicity 0.68→**0.86**, s_std 4.6→1.1, pred ÷16) yet nav ↓ |
| **Direct-metric (NO beam, descend quasimetric to goal)** | **45.0 %** | **0.244** | **best** — simplest |
| dream-subgoal (proper intermediate subgoal) | 42.0 % | 0.215 | committed/forward subgoal, still < direct |

**Three findings that survive scrutiny (slide-worthy):**
1. **The macro-option beam is dead weight.** `beam-match-A*` is stuck at **0.32–0.37 across every
   config**, independent of metric quality. Removing it (direct-metric) is best.
2. **Better metric ≠ better navigation.** The quasimetric made the distance much sharper (0.86
   monotone) but navigation got *worse* — proof the live bottleneck is the option/planning level,
   not the metric. And **deeper planning on a noisy cost actively hurts**.
3. **The subgoal mechanism's problem is placement, not stability.** We visualized goal+subgoal on
   every frame (cyan ring) + printed `subgoal@(x,y)`. A "farthest forward-progress + commit-until-
   reached" definition fixed the *teleporting* subgoal, but it still lands in **walled-off wrong
   regions** — because the quasimetric, while globally monotone, has **local errors** (says a
   region behind a wall is "near" the goal). The agent gets trapped there.

**So the useful hierarchy is the abstraction (z, ψ(z)) + quasimetric — NOT explicit subgoal
planning.** Best A\*-free result: **45 % / SPL 0.244**, all in latent space, 2 learned abstractions,
no A\* in the loop. The ceiling is the metric's *local* accuracy; the next real lever is a better
quasimetric (harder negatives / contrastive / bigger head), not more planning.

> ⚠️ **This §10 verdict is SUPERSEDED by §11.** "Explicit subgoal planning doesn't help" turned
> out to be an artifact of a *memoryless* low level — not a fact about hierarchy. Giving the low
> level a visit memory flips the result: the 2-level subgoal config goes 42 % → **84.5 %**, nearly
> double the flat direct-metric (45 %). The hierarchy was right; the controller was broken.

---

## 11. The breakthrough — low-level memory unlocks the hierarchy (84.5 → 85.5 %)

| config | success | SPL | note |
|---|---|---|---|
| direct-metric (flat, ex-best of §10) | 45.0 % | 0.244 | 1 abstraction, no subgoal |
| dream-subgoal, memoryless low level | 42.0 % | 0.215 | the §10 "subgoal doesn't help" |
| dream-subgoal + visit memory, replan ≥ MIN_COMMIT-gated junction | 84.5 / 85.0 % | 0.325 / 0.329 | jobs 77313/77378, seeds 0/1 |
| visit memory, **tiered** choice (metric decides among non-over-visited) | 53.5 % | 0.235 | job 77410 — **refuted**, see below |
| **visit memory + dream-subgoal recomputed EVERY step** | **85.5 / 86.0 %** | **0.341 / 0.357** | jobs 77435/77508, seeds 0/1 — **best** |
| every-step + coarse retrained with cross-traj hard negatives | 75.5 % | 0.287 | job 77379 — **refuted**, negatives hurt nav |

**~170/200 solved, A\*-free, zero extra training** (pure inference logic on the *existing*
checkpoints), still a strict 2-level H-JEPA in latent space. This nearly **doubles** the old
45 % ceiling and clears the organiser's ~66 % success (though their SPL 0.62 is more *efficient*
than our ~0.34 — we are more robust, they are more direct).

**Why the §10 verdict was wrong.** The subgoal was never dead weight — the **memoryless greedy low
level** was. It descends the quasimetric one step at a time with no record of where it has been, so
(a) it **orbits** local-minima basins of the noisy metric (the "57× revisits" seen in GIFs), and
(b) it **cannot back out of a dead-end**, because leaving one means temporarily *increasing* the
distance, which a pure greedy descender never chooses. With that low level, *any* high level looks
useless. Fix the low level and the subgoal suddenly pays.

**The two changes** (both training-free, in `eval_hjepa.py` only):
1. **Visit memory + revisit penalty.** Per-cell visit counts re-rank the candidate actions
   (`metric_rank + λ·visits[next_cell]`). The quasimetric stays the compass; revisits only nudge
   toward the unexplored frontier — which **breaks the orbits** and walks the agent out of
   dead-ends. Until now only *walls* were remembered (blocked-skip), never visited open cells.
2. **Subgoal cadence — recompute EVERY step (best).** We A/B'd three cadences on the same model:
   17-step over-commitment (misses turns), MIN_COMMIT-gated junction replan (84.5 %), and a
   *receding-horizon* subgoal redreamed every step. The last wins (**85.5 % / 0.341**): a fresh
   forward-progress waypoint each step both holds success and lifts SPL, because a too-sticky
   subgoal lets the agent loop. (Earlier we feared "teleporting" — but with the additive revisit
   memory underneath, recomputing every step is stable, not jittery.)

**A refuted variant (worth a slide — shows the metric is the real bottleneck).** We also tried
making the visit memory a *filter* instead of a penalty: among non-over-visited directions, let the
quasimetric alone pick (i.e. "follow the subgoal faithfully"). It **collapsed to 53.5 % / 0.235**.
Lesson: the strong, gradient-like revisit penalty is **load-bearing** — it compensates for the
metric's *local* errors by forcing exploration out of wrong basins. Following the (locally-wrong)
metric more faithfully just obeys a bad compass. So the deviations you see in the GIFs are not a
bug; they are the agent escaping a noisy metric. The lever for efficiency is therefore the **metric
itself** (the cross-trajectory-negatives retrain), not a more obedient controller.

**Honest caveats (say them on the slide):**
- **SPL ~0.34 is still low** — successful paths are ~2.6× optimal; the agent explores a lot. This,
  not success, is now the thing to improve, and it's a *metric* problem (see the refuted variant).
- **Confirmed across seeds, both configs** (200 mazes each, not a fluke): MIN_COMMIT cadence
  84.5 % / 0.325 (s0) · 85.0 % / 0.329 (s1); **every-step subgoal 85.5 % / 0.341 (s0) · 86.0 % /
  0.357 (s1)** — every-step is consistently +1 pt success and +0.02–0.03 SPL on both seeds, so it
  is the new best, robustly.

**We tried to crack the SPL ceiling via the metric — and it refused.** A coarse retrain with
**cross-trajectory hard negatives** (unreachable other-maze states pushed to large distance —
`train_coarse.py`, isolated `coarse_neg/`) was meant to sharpen the metric's *local* accuracy so
the agent wanders less. It **regressed to 75.5 % / 0.287** (job 77379). So the easy negatives
perturbed the embedding without fixing the local errors that matter — consistent with §10's theme
that this metric is *delicate*: changes that look principled (LayerNorm+quasimetric, harder
negatives) keep hurting navigation. The SPL ceiling (~0.34) stands; cracking it needs a smarter
metric objective (e.g. mined *within-maze* geodesic negatives), not the cheap cross-maze ones.

**Thesis for the talk:** a genuinely 2-level H-JEPA, A\*-free and all-latent, **beats the flat
controller (45 % → 84.5 %)** once the low level is given the one thing every embodied agent has —
memory of where it has been.

---

## Appendix — artifacts

**Code (ours):**
- `eb_jepa/hjepa.py` — `dream_macro_option`, `CoarseEncoder`, `CoarsePredictor`,
  `ema_update`, `coarse_jepa_loss`, `coarse_beam`, `pick_fine_action` (9/9 unit tests in
  `tests/test_hjepa.py`)
- `examples/ac_video_jepa/maze/train_coarse.py` — coarse JEPA trainer (dreams + distance)
- `examples/ac_video_jepa/maze/eval_hjepa.py` — A\*-free 2-level eval (seeded, SPL, W&B)
- `examples/ac_video_jepa/maze/diag_hjepa.py` — coarse-space navigability diagnostic
- `rerun_hjepa.sh` / `diag_hjepa.sh` — sbatch pipelines

**Specs / plans:** `docs/superpowers/specs/2026-06-20-hjepa-two-level-design.md`,
`docs/superpowers/plans/2026-06-20-hjepa-two-level.md` (and the superseded 1-level quasimetric design).

**Key numbers:** baseline reproduced 81–90 % (32 mazes, unseeded); 2-level diagnostic
monotonicity 0.68 → 0.86 (quasimetric+LayerNorm); 2-level success 0 % → 45 % (wall-avoid) →
**84.5 % / SPL 0.325** (low-level visit memory + cadence, job 77313, 200 seeded mazes). Maze 21×21, img 63.

**W&B:** https://wandb.ai/tristan-faure-epita/eb_jepa — group `hjepa`.
