# Maze — AC-Video-JEPA world model + A*-free hierarchical navigation

This is the **entry point** for the maze use-case (the other use-case of
`ac_video_jepa` is **Two Rooms**, see `../README.md`). The maze reuses the exact same
stack — an Impala encoder, an `RNNPredictor`, `JEPA.unroll`, and the MPPI planner —
on a procedurally generated grid maze, and adds a learned hierarchy so the agent
can navigate **without A\* at evaluation**.

## The progression (read in this order)

| stage | what | needs A\* at eval? | success (32 mazes) |
|---|---|---|---|
| **Baseline** | world model + MPPI planning toward **A\* waypoints**; cost is swappable (distance vs **learned TD-MPC value**) | yes (waypoints) | value **37.5 %** vs distance 6.25 % (greedy-global 0 %) |
| **Level 1** | **A\*-free**: a learned `SubgoalPredictor` replaces A\* + a low-level lookahead reacher | **no** | **65.6 %**, SPL 0.62 |
| **Level 2** | **hierarchization**: co-train the two levels (shared latent) | **no** | 46.9 % (did not beat L1 — see below) |
| *control* | random-walk baseline (same budget) | no | 0 % |

A\* is used **only as a training teacher** (it generates the data and supervises the
subgoal predictor) and to **size the eval step budget** — never in the agent's
decision loop. Everything is selected by config flags (see §6, *modular features*).

---

## Entry points — which `main` to run

The maze use-case has **three training entry points** (one shared + two maze-specific) and
**three evals**. Everything is `python -m <module>`; checkpoints and `out_dir`s are your choice.

| kind | `python -m …` | role | key args |
|---|---|---|---|
| **train** | `examples.ac_video_jepa.main` | **fine world model** — the base maze WM (shared trainer, `env_name: maze`) | `--fname examples/ac_video_jepa/cfgs/train/maze/train_maze_aux.yaml --meta.model_folder=<dir>` |
| **train** | `examples.ac_video_jepa.maze.main_subgoal` | **Level 1** — train the `SubgoalPredictor` (A\*-free high level), WM frozen | `<fine_ckpt> <out_dir> <N> <epochs>` |
| **train** | `examples.ac_video_jepa.maze.main_cotrain` | **Level 2** — co-train both levels on a shared latent | `<fine_ckpt> <subgoal_ckpt> <out_dir> <N> <epochs> <freeze_ep> <enc_lr>` |
| eval | `examples.ac_video_jepa.main` *(+`--meta.eval_only_mode=True`)* | **baseline planning** eval of the fine WM (MPPI + chosen objective) | `--fname … --eval.plan_cfg_path=… --eval.eval_cfg_path=…` |
| eval | `examples.ac_video_jepa.maze.eval_subgoal` | **A\*-free** closed-loop eval (Levels 1 & 2; SPL + GIFs) | `<fine_ckpt> <subgoal_ckpt> <out_dir> <num_ep> <lookahead> <revisit_pen> <n_gifs> <budget_factor> <margin>` |
| eval | `examples.ac_video_jepa.maze.eval_random` | random-walk **control** baseline (sanity floor) | `<fine_ckpt> <out_dir> <num_ep> <budget_factor> <margin>` |

**Pipeline:** `main` (fine WM) → `main_subgoal` (L1) → *optionally* `main_cotrain` (L2), each
followed by `eval_subgoal`. Steps are detailed in §2–§6 below. (`plots_maze_value.py` just
renders the baseline value-vs-distance chart.)

---

## 1. Data — generated **online** (no download)
Each episode is a fresh random maze: a DFS generator builds the walls, **A\* solves
the shortest path** start→goal, and the path (with `wall_bump_prob` collisions, so
the world model learns walls) becomes the trajectory. Nothing is stored on disk.
- code: `eb_jepa/datasets/maze/` (`gpu_generator.py`, `maze_solver.py`, `env.py`,
  `maze_dataset.py`, `normalizer.py`), geometry in `data_config.yaml`.
- dispatched by `env_name: maze` in `eb_jepa/datasets/utils.py` (Two Rooms uses
  `env_name: two_rooms` — same dispatch, untouched).

## 2. Train the fine world model
Standard AC-Video-JEPA training (encoder + RNN predictor + VC/IDM regularizer +
position probe). The three keys that made maze planning work: **snap** actions to
the grid, an **aux-position** loss (`aux_pos_coeff`) so the latent is
position-decodable, and **`wall_bump_prob`** so the model learns collisions.
```bash
python -m examples.ac_video_jepa.main --fname examples/ac_video_jepa/cfgs/train/maze/train_maze_aux.yaml \
    --meta.model_folder=$EBJEPA_CKPTS/maze/exp_value
```
Configs: `train_maze.yaml` (base), `train_maze_aux.yaml` (proven, aux-pos),
`train_maze_{small,big,long}.yaml` (size/horizon variants).

## 3. Evaluate (planning)
Eval-only mode loads a checkpoint and runs the MPPI planner with a chosen
**objective** (the planning cost) and **eval** config:
```bash
python -m examples.ac_video_jepa.main --fname examples/ac_video_jepa/cfgs/train/maze/train_maze_value.yaml \
    --meta.load_model=True --meta.eval_only_mode=True --meta.skip_unroll_eval=True \
    --eval.plan_cfg_path=examples/ac_video_jepa/cfgs/planning/maze/planning_mppi_value_wp2_pl4.yaml \
    --eval.eval_cfg_path=examples/ac_video_jepa/cfgs/eval/maze/eval_maze_med.yaml
```

## 4. Baseline — planning with A\* waypoints (swappable cost)
The MPC **objective** is config-selected via `objective_name_map` (`planning.py`):
`repr_dist` (latent MSE), `probe_pos` (decoded-position distance), or
**`learned_value`** (a TD-MPC value head `V(z, z_goal)` trained by TD on the world
model's own rollouts — `value_coeff`/`freeze_world_model` in `train_maze_value.yaml`).
Result: with A\* waypoints the **learned value 37.5 %** beats the distance cost
6.25 %; greedy-global is 0 % for all. → **`README_value.md`**.

## 5. Level 1 — A\*-free navigation (learned subgoals)
- **High level** `SubgoalPredictor(z, goal) → next waypoint` (`eb_jepa/hierarchical.py`),
  trained supervised on A\* waypoints (`main_subgoal.py`).
- **Low level**: reach the waypoint with the frozen, wall-aware fine WM via a
  **K-step lookahead** reacher + execution-feedback blocked-skip (`eval_subgoal.py`).
```bash
python -m examples.ac_video_jepa.maze.main_subgoal  <fine_ckpt> <out_dir> 4 12      # N=4, 12 epochs
python -m examples.ac_video_jepa.maze.eval_subgoal  <fine_ckpt> <out_dir>/subgoal.pth.tar \
       results/maze_subgoal 32 4 0.05 32 4 10   # num_ep lookahead revisit_pen n_gifs budget_factor margin
```
Result: **65.6 % success / SPL 0.62**, A\*-free. → **`README_hierarchical.md`**.

<p align="center">
  <em>Running <code>eval_subgoal.py</code> with <code>n_gifs &gt; 0</code> writes per-episode GIFs of
  A*-free navigation into your <code>out_dir</code> — the agent reaches the goal with no A* in the
  decision loop (learned subgoals + lookahead reacher).</em>
</p>

## 6. Level 2 — hierarchization (co-training)
Jointly fine-tune encoder + predictor + probe + subgoal on a **shared latent**
(staged unfreeze, gentle encoder LR) — `main_cotrain.py`:
```bash
python -m examples.ac_video_jepa.maze.main_cotrain <fine_ckpt> <subgoal_ckpt> <out_dir> 4 8 2 5e-5
python -m examples.ac_video_jepa.maze.eval_subgoal <out_dir>/latest.pth.tar <out_dir>/subgoal.pth.tar \
       results/maze_cotrain 32 4 0.05 32 4 10
```
Honest result: co-training **lowers the subgoal loss but does not beat L1** (46.9 %):
moving the encoder erodes the fragile wall-aware fine WM the low level relies on.
**Freeze the WM + invest in the planner.** Control: `eval_random.py` → 0 %.

## 7. Modular features (every knob is a config/flag)
| feature | flag / arg | where |
|---|---|---|
| environment | `env_name: maze` | `datasets/utils.py` |
| maze geometry | `maze_height/width`, `cell_size`, `min_path_length`, `wall_bump_prob` | `cfgs/train/maze/train_maze*.yaml`, `data_config.yaml` |
| planning cost | `planning_objective.objective_type` ∈ {`repr_dist`,`repr_dist_collision`,`probe_pos`,`learned_value`} | `cfgs/planning/maze/planning_*.yaml` → `objective_name_map` |
| TD-MPC value head | `value_coeff`, `value_gamma`, `value_lr`, `freeze_world_model` | `main.py` |
| aux-position | `aux_pos_coeff` | `main.py`, `cfgs/train/maze/train_maze*.yaml` |
| action-snap / waypoints | `snap_actions_to_grid`, `waypoint_mode` | `eb_jepa/planning.py`, `cfgs/planning/maze/planning_*.yaml` |
| subgoal horizon | `N` (cells ahead) | `main_subgoal.py` arg |
| low-level reacher | `lookahead K`, `revisit_pen` | `eval_subgoal.py` args |
| eval step budget | `budget_factor`·A\* + `margin` (A\* sizes the clock only) | `eval_subgoal.py` args |
| co-training | `freeze_epochs`, `enc_lr` (staged unfreeze) | `main_cotrain.py` args |

## 8. File map
```
eb_jepa/
  datasets/maze/                 # online generation (DFS + A* solve), env, normalizer
  planning.py                    # GCAgent + MPPI + objective_name_map (repr_dist/probe_pos/learned_value)
  state_decoder.py               # MLPXYHead (position probe) + GoalValueHead (TD-MPC value)
  hierarchical.py                # SubgoalPredictor (high level) + fine_kstep_target (lookahead)
examples/ac_video_jepa/
  main.py / eval.py              # SHARED trainer/eval (env-selected: two_rooms | maze)
  cfgs/                          # SHARED config tree (sibling envs under each stage)
    train/maze/                  #   train_maze*.yaml
    eval/maze/                   #   eval_maze*.yaml
    planning/maze/               #   planning_mppi_*.yaml
                                 #   (base planning_mppi.yaml lives in cfgs/planning/two_rooms/)
  maze/                          # <- this folder (maze scripts, READMEs)
    maze_fine_wm.py              #   build_fine(): rebuild frozen fine WM for inference
    main_subgoal.py / eval_subgoal.py   # Level 1: A*-free subgoal nav (+ SPL, GIFs)
    main_cotrain.py              #   Level 2: co-training (shared latent)
    eval_random.py               #   random-walk control
    plots_maze_value.py          #   baseline value-vs-distance bar chart
    README.md (this) · README_value.md · README_hierarchical.md
Eval/plot scripts write their outputs to whatever out_dir you pass on the CLI.
```
```bash
# random control (same budget) — sanity floor
python -m examples.ac_video_jepa.maze.eval_random <fine_ckpt> results/maze_random_baseline 32 4 10
```
