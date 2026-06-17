# Hierarchical maze JEPA — A*-FREE navigation (trained with A*, evaluated without)

**Goal:** make the maze solvable at eval with **NO A\*** (no waypoints, no A* prior,
no A* fallback in the decision loop), via a two-level hierarchy. A* is used **only
as a training teacher** and to *size the eval clock* — never to choose actions.

> **Headline.** Frozen world model + a learned **SubgoalPredictor** + a K-step
> lookahead reacher solves 21×21 mazes **A*-FREE at ~66 % success, SPL 0.62**
> (32 held-out mazes, difficulty-proportional budget), vs **0 %** for greedy
> planning without the hierarchy.

## 1. Why a hierarchy (the 0 % problem)

Planning straight to a far global goal fails (**0 %**): the learned value / planner
horizon can't span a 50-cell maze (a flat-saturated value gives MPPI no gradient).
The fix is a **two-level** decomposition (feudal / closed-loop subgoals):

- **High level — `SubgoalPredictor(z_current, goal_xy) → next waypoint position`**
  (`eb_jepa/hierarchical.py`). The state latent encodes the *whole* maze (the wall
  mask is in the obs image) + the agent position, so it can learn to route. Trained
  **supervised on A\* trajectories** (label = the A* position `N=4` cells ahead).
  At eval it proposes the waypoints itself.
- **Low level — reach the waypoint** with the **frozen fine world model**, which is
  *wall-aware* (trained with `wall_bump`, so it predicts "stay" into a wall). At
  each step we roll the fine WM **K steps per cardinal** (lookahead), pick the
  direction whose probe-decoded endpoint is closest to the waypoint, with
  **execution-feedback blocked-skip** (a direction that doesn't move the agent is
  blacklisted at that cell) + a revisit penalty + no-immediate-U-turn. Closed-loop:
  the subgoal is re-predicted every step.

## 2. Eval protocol (how "A*-free" is measured)

- **A*-free decisions.** The agent only ever sees the current obs + the global goal.
  No waypoints, prior, or fallback from A* enter the action loop.
- **No oracle on feasible moves.** The agent freely picks any of the 4 cardinals; it
  *can* attempt a wall-hitting move. Walls are handled without any oracle: (a) the
  **wall-aware fine-WM lookahead** predicts a blocked move as "stay" → low score →
  avoided; (b) if it still bumps, the env keeps it in place (no pass-through) and the
  **execution-feedback blocked-skip** ("did I move?") tries another direction. A
  bumped move still **costs a budget step** (realistic). No A*, no feasible-move list.
- **Difficulty-proportional budget.** Each episode's step budget is
  `⌈factor · len(A*) + margin⌉` (default **4·A\*+10**). A* is used *only* to set this
  time limit (a fixed 180 unfairly failed long mazes — e.g. one with a 120-cell A*
  path needs ~490 steps); it never guides the agent. Standard maze-RL convention.
- **Metrics.** *Success rate* (reached the goal cell within budget) **and SPL**
  (Success weighted by Path Length = `success · len(A*) / max(agent_moves, len(A*))`,
  i.e. efficiency vs the optimal path).
- **Held-out mazes.** Data is generated **online** (DFS maze + A* solve), so the
  32 eval mazes are fresh — never seen in training.

```bash
# train high level (frozen WM), then A*-free eval with budget+SPL+GIFs:
python -m examples.ac_video_jepa.maze.main_subgoal <fine_ckpt> <out_dir> 4 12
python -m examples.ac_video_jepa.maze.eval_subgoal <fine_ckpt> <out_dir>/subgoal.pth.tar <out_dir> \
       32 4 0.05 32 4 10     # num_ep, lookahead K, revisit_pen, n_gifs, budget_factor, margin
```

## 3. Files
- `eb_jepa/hierarchical.py` — `SubgoalPredictor` (high level) + `fine_kstep_target` (low-level K-step lookahead)
- `maze_fine_wm.py` — `build_fine()`: rebuild the frozen fine world model for inference
- `main_subgoal.py` — train the SubgoalPredictor (supervised on A* waypoints, frozen fine WM)
- `eval_subgoal.py` — A*-free closed-loop eval: K-step lookahead reacher, A*-proportional budget, SPL, per-episode GIF dump
- `main_cotrain.py` — joint shared-latent fine-tuning (staged unfreeze) phase
- `eval_random.py` — random-walk control baseline

See `README.md` for the full baseline → Level 1 → Level 2 overview and the
modular-features table.

## 4. Results (32 held-out 21×21 mazes, **zero A\* in the decision loop**)

| config | budget | success | SPL |
|---|---|---|---|
| greedy, no hierarchy | fixed | **0 %** | — |
| random-persist baseline | 4·A\*+10 | **0 %** | 0.000 |
| subgoal (frozen), planner 1-step | fixed 180 | 31.25 % | — |
| **subgoal (frozen) + lookahead K=4** | **4·A\*+10** | **65.62 %** | **0.616** |
| co-training (staged) + lookahead K=4 | 4·A\*+10 | 46.88 % | 0.453 |
| co-training (staged) + planner K=4 | fixed 180 | 40.62 % | — |
| co-training naïf (enc_lr 2e-4), 1-step | fixed 180 | 12.50 % | — |

**Progression: 0 % → 31 % (hierarchy) → ~66 % (K-step lookahead + fair budget).**

**What worked.** (1) The **hierarchy** turns 0 % → 31 % A*-free (learned subgoals
replace A* routing). (2) The **K-step lookahead reacher** (anti-myopia + dead-end
avoidance) is the big lever, ~31 % → ~66 %. (3) A **fair, difficulty-proportional
budget** removes the long-maze failures.

**Co-training: a clean negative.** Jointly fine-tuning encoder+predictor+subgoal
(shared latent) *lowers the subgoal MSE* (0.047 < 0.059) **but does not improve —
and tends to reduce — success**: moving the encoder **erodes the fragile wall-aware
fine WM** the low level depends on (it was trained with `wall_bump`). The naïve
co-train (enc_lr 2e-4, 1-step planner) collapsed to 12.5 %; the **repaired** version
(staged unfreeze + enc_lr 5e-5) recovers to 40.6 % (fixed budget) but still trails
the **frozen WM** config. Lesson: **the wall-aware world model is too precious to
move — freeze it and invest in the planner.**

`eval_subgoal.py` writes per-episode GIFs of A*-free navigation (`<out_dir>/ep*_succ.gif`,
plus failures) and JSON metrics (`<out_dir>/subgoal_eval.json`) into the `out_dir` you pass.
