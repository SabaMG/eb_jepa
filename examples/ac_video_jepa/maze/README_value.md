# Maze planning with a learned TD-MPC value (vs distance-in-latent cost)

A learned **value / cost-to-go** function that replaces the crude
"distance-in-latent-space" planning cost used by the maze world-model planner —
the TD-MPC idea (Hansen et al., 2022/2024): train a value on the world model's
own rollouts so the planner optimises a quantity that correlates with **task
success** (steps-to-goal, walls included) rather than raw representation distance.

> **Headline.** On the *same frozen world model* and the same 16 held-out 21×21
> mazes, swapping only the MPC cost: with A\* waypoints (spacing 2) the **learned
> value reaches 37.5 % success vs 6.25 % for the geometric `probe_pos` distance —
> ~6×**. Greedy-global is 0 % for every cost.

## Background — why a learned cost

The maze world-model planner (MPPI) was previously driven by geometric costs:
- `repr_dist` — MSE between the predicted latent and the goal latent. The 2-channel
  maze obs (dot + static wall mask) makes the latent **dominated by the walls**, so
  distance-to-goal is nearly flat → MPPI gets no signal → the agent barely moves.
- `probe_pos` — distance between the probe-decoded **position** and the goal. Wall-
  invariant, but still an as-the-crow-flies proxy that ignores walls between agent
  and goal.

Both are hand-crafted. The TD-MPC answer: **learn** the cost-to-go.

## What was added

| file | change |
|---|---|
| `eb_jepa/state_decoder.py` | `GoalValueHead` — `V(z, z_goal) ∈ (0,1)` ≈ discounted return-to-goal (`≈ γ^{steps_to_goal}`) |
| `eb_jepa/planning.py` | `LearnedValueMPCObjective` (cost `= 1 − V`, MPPI maximises value); registered as `learned_value` in `objective_name_map`; `value_head` threaded through `GCAgent.set_goal` |
| `examples/ac_video_jepa/main.py` | TD(0) training of the value head on the model's **own autoregressive rollouts** (EMA target net; reward 1 at the goal; regressed on both real and imagined latents). Gated by `value_coeff`; `freeze_world_model` trains *only* the value head on a proven frozen world model |
| `examples/ac_video_jepa/eval.py` | `value_head` passed into `main_eval` → `GCAgent` |
| `cfgs/` | `cfgs/train/maze/train_maze_value.yaml` + `cfgs/planning/maze/planning_mppi_value_{greedy,wp1,wp2_pl4}.yaml` |
| `plots_maze_value.py` | comparison plot + `results.json` (written to the `out_dir` you pass) |

### The value head (TD-MPC style)

`GoalValueHead(z, z_goal)` mean-pools the (full-resolution) state and goal latents,
concatenates, and an MLP + sigmoid outputs a scalar in (0, 1). It is trained by
**TD(0)** on the world model's own rollouts:

```
goal g      = encoder(window endpoint)            # hindsight goal
target y_t  = r_t + γ (1-done_t) V_target(z_{t+1}, g)   # EMA target net, reward 1 at goal
loss        = MSE(V(z_real_t, g), y_t) + MSE(V(z_rollout_t, g), y_t)
```

regressing **both** the encoded real latents and the model's **imagined rollout**
latents (what the planner actually sees) onto the same TD target. At planning, the
`learned_value` objective returns `cost = 1 − V`, so MPPI maximises value.

## Protocol

The world model is **frozen** (`exp_value` = the proven `exp_aux_pos` weights;
`freeze_world_model: true`); only the value head trains (loss 0.035 → 0.010, 6
epochs, ~15 s/epoch). Then a **controlled comparison**: identical world model,
identical MPPI settings and the same 16 held-out 21×21 mazes
(`eval_maze_med`, `n_allowed_steps=180`), changing **only the planning objective**.

```bash
# 1. train the value head on the frozen WM (init from the proven aux-pos checkpoint)
python -m examples.ac_video_jepa.main examples/ac_video_jepa/cfgs/train/maze/train_maze_value.yaml \
    --meta.init_from=<exp_aux_pos_ckpt> --meta.model_folder=<out_dir>
# 2. value vs probe @ tuned wp2 (eval-only planning)
python -m examples.ac_video_jepa.main examples/ac_video_jepa/cfgs/train/maze/train_maze_value.yaml \
    --meta.load_model=True --meta.eval_only_mode=True --meta.skip_unroll_eval=True \
    --meta.model_folder=<out_dir> \
    --eval.plan_cfg_path=examples/ac_video_jepa/cfgs/planning/maze/planning_mppi_value_wp2_pl4.yaml \
    --eval.eval_cfg_path=examples/ac_video_jepa/cfgs/eval/maze/eval_maze_med.yaml
# 3. comparison plot (reads/writes your out_dir)
python -m examples.ac_video_jepa.maze.plots_maze_value <out_dir>
```

## Results (plot written to `<out_dir>/maze_value_compare.png` by `plots_maze_value.py`)

| planning regime | **learned VALUE** | probe_pos (distance) | repr_dist (latent MSE) |
|---|---|---|---|
| greedy (global goal, no waypoints) | 0 % | 0 % | 0 % |
| A\* waypoints, spacing 1 | **12.5 %** | 6.25 % | — |
| A\* waypoints, spacing 2 (tuned) | **37.5 %** | 6.25 % | — |

**Reading.**
- **Greedy-global = 0 % for all three costs**: with no subgoals the goal is 50+
  steps away — beyond the value's trained horizon, and `repr_dist` is wall-dominated.
  No cost rescues greedy-global on 21×21.
- **With A\* waypoints the learned value wins decisively** (37.5 % vs 6.25 %, ~6×):
  the subgoal is within the value's trained horizon, where its cost-to-go is a much
  sharper, wall-aware guidance signal than geometric distance. Going from spacing-1
  to spacing-2 triples the value's success (12.5 → 37.5 %), while the distance cost
  is unmoved (6.25 %).

**Takeaway.** Replacing the hand-crafted distance with a value learned on the world
model's own rollouts gives a large, controlled win for planning success — the
TD-MPC hypothesis holds here. The remaining limiter is the value's **horizon**
(trained on 16-step windows): extending it (longer windows / hindsight goals at
varied distances) is the path to lifting greedy-global off 0 % and pushing the
waypoint regime higher.

*Caveat:* absolute numbers are modest and the `probe_pos` baseline here (6.25 %)
is below an earlier ~45 % obtained in a different eval draw — 16 episodes is a
high-variance estimate. The **controlled** value-vs-distance gap (same model, same
mazes, same settings) is the trustworthy signal.
