# Latent Subgoal Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the A\*-cloning `SubgoalPredictor` + constant-action reacher with a learned wall-aware latent quasimetric `d(z_a, z_b)` that drives both subgoal selection and the low-level reacher on the frozen world model — Quasimetric RL on a frozen JEPA WM.

**Architecture:** One new trained head, a **MRN quasimetric** `d(z_a,z_b) = ‖f(a)−f(b)‖₂ + maxᵢ relu(g(b)ᵢ−g(a)ᵢ)` (triangle-inequality guaranteed, dependency-free; IQE is an optional swap). The WM stays frozen. A guided beam over WM rollouts proposes reach-bounded candidate latents; the subgoal is the candidate minimizing `d(·, z_goal)`; the low level descends `d` toward it. No A\* in the decision loop.

**Tech Stack:** PyTorch, `uv` (run tests with `uv run pytest`), the existing `eb_jepa` package (`JEPA`, `ImpalaEncoder`, `RNNPredictor`, `build_fine`, maze env + data pipeline), SLURM/sbatch on Dalia for GPU training/eval.

**Spec:** `docs/superpowers/specs/2026-06-20-latent-subgoal-planning-design.md`

**Where things run:** Pure-logic units (head, pair sampling, loss assembly, planner selection) are CPU torch — **tested locally** with `uv run pytest`. The actual distance training, validation gate, and maze eval need the GPU + data pipeline — **run on Dalia via sbatch** (the laptop has no GPU/data).

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `eb_jepa/state_decoder.py` | Modify | add `MRNDistanceHead` next to `GoalValueHead` |
| `eb_jepa/distance.py` | Create | pure helpers: `sample_pairs`, `distance_training_loss`, `monotonic_fraction` |
| `eb_jepa/hierarchical.py` | Modify | add planner: `beam_candidates`, `select_subgoal`, `pick_action`, `latent_subgoal_step` |
| `examples/ac_video_jepa/maze/train_distance.py` | Create | train the head on the frozen WM (cluster) |
| `examples/ac_video_jepa/maze/validate_distance.py` | Create | §8 monotonicity gate (cluster) |
| `examples/ac_video_jepa/maze/eval_latent_subgoal.py` | Create | A\*-free closed-loop eval with the new planner (cluster) |
| `rerun_latent_subgoal.sh` | Create | sbatch: train → validate → eval |
| `tests/test_quasimetric.py` | Create | head properties (non-neg, identity, triangle, shape) |
| `tests/test_distance_helpers.py` | Create | `sample_pairs`, `distance_training_loss`, `monotonic_fraction` |
| `tests/test_latent_subgoal_planner.py` | Create | `select_subgoal`, `pick_action`, `beam_candidates` with stubs |

---

## Task 1: MRN quasimetric distance head

**Files:**
- Modify: `eb_jepa/state_decoder.py`
- Test: `tests/test_quasimetric.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quasimetric.py
import torch
from eb_jepa.state_decoder import MRNDistanceHead


def _head():
    torch.manual_seed(0)
    return MRNDistanceHead(input_shape=8, embed=16, asym=16)


def test_output_shape_and_nonneg():
    head = _head()
    za = torch.randn(5, 8, 1, 1, 1)
    zb = torch.randn(5, 8, 1, 1, 1)
    d = head(za, zb)
    assert d.shape == (5,)
    assert (d >= 0).all()


def test_identity_is_zero():
    head = _head()
    z = torch.randn(4, 8, 1, 1, 1)
    d = head(z, z)
    assert torch.allclose(d, torch.zeros(4), atol=1e-5)


def test_triangle_inequality():
    head = _head()
    a = torch.randn(32, 8, 1, 1, 1)
    b = torch.randn(32, 8, 1, 1, 1)
    c = torch.randn(32, 8, 1, 1, 1)
    dac = head(a, c)
    dab = head(a, b)
    dbc = head(b, c)
    assert (dac <= dab + dbc + 1e-4).all()


def test_accepts_pooled_time_dim():
    head = _head()
    za = torch.randn(3, 8, 2, 1, 1)   # T=2 gets pooled
    zb = torch.randn(3, 8, 2, 1, 1)
    assert head(za, zb).shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quasimetric.py -v`
Expected: FAIL with `ImportError: cannot import name 'MRNDistanceHead'`

- [ ] **Step 3: Write minimal implementation**

Append to `eb_jepa/state_decoder.py`:

```python
class MRNDistanceHead(nn.Module):
    """Metric Residual Network quasimetric d(z_a, z_b) >= 0 with a guaranteed
    triangle inequality (Liu et al., 2022):

        d(a, b) = ||f(a) - f(b)||_2  +  max_i relu( g(b)_i - g(a)_i )

    The first term is a symmetric metric, the second an asymmetric quasimetric;
    their sum is a quasimetric (d(a,a)=0, triangle inequality holds). Interprets
    `d` as steps-to-go between two latents under the frozen world model.

    Pooled-latent interface like GoalValueHead: input latents are [N, C, *spatial]
    (e.g. [N, C, 1, 1, 1] from the impala encoder); dims >= 2 are mean-pooled.
    """

    def __init__(self, input_shape, embed=256, asym=256, hidden=512):  # input_shape = C
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(input_shape, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, embed),
        )
        self.g = nn.Sequential(
            nn.Linear(input_shape, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, asym),
        )

    @staticmethod
    def _pool(z):
        # [N, C, *spatial] -> [N, C]
        if z.dim() <= 2:
            return z
        return z.flatten(2).mean(dim=-1)

    def forward(self, z_a, z_b):
        a = self._pool(z_a)
        b = self._pool(z_b)
        sym = torch.linalg.norm(self.f(a) - self.f(b), dim=-1)          # [N]
        asym = torch.relu(self.g(b) - self.g(a)).amax(dim=-1)           # [N]
        return sym + asym
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_quasimetric.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add eb_jepa/state_decoder.py tests/test_quasimetric.py
git commit -m "feat(distance): MRN quasimetric latent distance head"
```

---

## Task 2: Distance training helpers (pair sampling + loss)

**Files:**
- Create: `eb_jepa/distance.py`
- Test: `tests/test_distance_helpers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distance_helpers.py
import torch
from eb_jepa.distance import sample_pairs, distance_training_loss, monotonic_fraction


def test_sample_pairs_valid():
    g = torch.Generator().manual_seed(0)
    i, j = sample_pairs(T=10, num_pairs=200, generator=g)
    assert i.shape == (200,) and j.shape == (200,)
    assert (i < j).all()
    assert (i >= 0).all() and (j <= 9).all()


def test_distance_training_loss_scalar_and_grad():
    torch.manual_seed(0)
    # tiny head stub: d = sum of pooled |a-b|  (differentiable, non-negative)
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(()))
        def forward(self, za, zb):
            a = za.flatten(2).mean(-1) if za.dim() > 2 else za
            b = zb.flatten(2).mean(-1) if zb.dim() > 2 else zb
            return self.w * (a - b).abs().sum(-1)
    head = Stub()
    z = torch.randn(4, 8, 10, 1, 1)          # [B, C, T, h, w]
    i, j = sample_pairs(T=10, num_pairs=16)
    loss = distance_training_loss(head, z, i, j)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert head.w.grad is not None


def test_monotonic_fraction():
    # strictly decreasing -> 1.0 ; flat/increasing -> lower
    dec = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    inc = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert monotonic_fraction(dec) == 1.0
    assert monotonic_fraction(inc) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_distance_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eb_jepa.distance'`

- [ ] **Step 3: Write minimal implementation**

```python
# eb_jepa/distance.py
"""Self-supervised training helpers for the latent quasimetric distance head.

`d(z_a, z_b)` is regressed to the temporal gap j-i between two latents on a
trajectory. On A* (shortest) paths the gap is the exact step-distance, so A* is
only a label oracle in the DATA -- never a planner. See the design spec
docs/superpowers/specs/2026-06-20-latent-subgoal-planning-design.md.
"""
import torch
import torch.nn.functional as F


def sample_pairs(T, num_pairs, generator=None):
    """Sample (i, j) index pairs with 0 <= i < j <= T-1, vectorized."""
    i = torch.randint(0, T - 1, (num_pairs,), generator=generator)
    span = (T - 1 - i).to(torch.float32)                      # >= 1
    off = (torch.rand(num_pairs, generator=generator) * span).long() + 1
    j = torch.clamp(i + off, max=T - 1)
    return i, j


def distance_training_loss(d_head, z, i, j):
    """Huber regression of d(z_i, z_j) onto the temporal gap (j - i).

    z: [B, C, T, h, w] latents (frozen-WM encoded). i, j: [P] long index tensors.
    Returns a scalar loss. Pairs are shared across the batch (b-major flatten).
    """
    B, C, T, h, w = z.shape
    P = i.shape[0]
    zi = z.index_select(2, i.to(z.device)).permute(0, 2, 1, 3, 4).reshape(B * P, C, 1, h, w)
    zj = z.index_select(2, j.to(z.device)).permute(0, 2, 1, 3, 4).reshape(B * P, C, 1, h, w)
    label = (j - i).to(z.device, torch.float32).repeat(B)     # [B*P], b-major
    pred = d_head(zi, zj)                                      # [B*P]
    return F.smooth_l1_loss(pred, label)


def monotonic_fraction(seq):
    """Fraction of consecutive steps where `seq` strictly decreases. seq: [L]."""
    if seq.numel() < 2:
        return 1.0
    diffs = seq[1:] - seq[:-1]
    return float((diffs < 0).float().mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_distance_helpers.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add eb_jepa/distance.py tests/test_distance_helpers.py
git commit -m "feat(distance): temporal-gap pair sampling, Huber loss, monotonicity metric"
```

---

## Task 3: Planner — candidate generation and selection (pure logic)

**Files:**
- Modify: `eb_jepa/hierarchical.py`
- Test: `tests/test_latent_subgoal_planner.py`

These three functions are the planner's testable core. `beam_candidates` rolls the
frozen WM; `select_subgoal` picks the reach-bounded goal-closest candidate;
`pick_action` is the 1-step low-level descent. All take the WM/head as args so they
test with stubs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_latent_subgoal_planner.py
import torch
from eb_jepa.hierarchical import beam_candidates, select_subgoal, pick_action


class StubJEPA:
    """Latent = a 2D 'position' in channels [0,1]; predictor adds the action.
    z shape [N, C, 1, 1, 1], action a shape [N, 2, 1] (dir * cell_size)."""
    def predictor(self, z, a):
        out = z.clone()
        out[:, 0, 0, 0, 0] += a[:, 0, 0]
        out[:, 1, 0, 0, 0] += a[:, 1, 0]
        return out


def _pos_distance_head(goal_xy):
    # d(z, _) = || pos(z) - goal_xy ||  (ignores second arg; used as score_fn)
    def d(z, _zb=None):
        pos = z[:, :2, 0, 0, 0]
        return torch.linalg.norm(pos - goal_xy, dim=-1)
    return d


def test_beam_candidates_shapes():
    jepa = StubJEPA()
    z0 = torch.zeros(1, 4, 1, 1, 1)
    goal = torch.tensor([3.0, 0.0])
    score = lambda z: _pos_distance_head(goal)(z)
    zc, first, depth = beam_candidates(jepa, z0, score, horizon=3, beam_width=4, cell_size=1.0)
    assert zc.shape[0] == first.shape[0] == depth.shape[0]
    assert depth.min() >= 1 and depth.max() <= 3
    assert set(first.tolist()).issubset({0, 1, 2, 3})


def test_select_subgoal_picks_min_score_above_dmin():
    # 3 candidates: scores [0.1, 0.9, 0.2], depths [1, 3, 3]; d_min=2
    zc = torch.zeros(3, 4, 1, 1, 1)
    scores = torch.tensor([0.1, 0.9, 0.2])
    depth = torch.tensor([1, 3, 3])
    idx = select_subgoal(scores, depth, d_min=2)
    assert idx == 2  # 0.1 excluded (depth<2); 0.2 < 0.9


def test_pick_action_moves_toward_subgoal():
    jepa = StubJEPA()
    z_t = torch.zeros(1, 4, 1, 1, 1)
    z_sg = torch.zeros(1, 4, 1, 1, 1)
    z_sg[0, 0, 0, 0, 0] = 1.0   # subgoal is +1 in x  (CARDINALS row 2 = [0,1]? see note)
    # d = latent L2 to z_sg
    d_head = lambda za, zb: torch.linalg.norm(
        (za[:, :2, 0, 0, 0] - zb[:, :2, 0, 0, 0]), dim=-1)
    a = pick_action(jepa, d_head, z_t, z_sg, cell_size=1.0)
    # CARDINALS = [[1,0],[-1,0],[0,1],[0,-1]]; +x is row 0
    assert a == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_latent_subgoal_planner.py -v`
Expected: FAIL with `ImportError: cannot import name 'beam_candidates'`

- [ ] **Step 3: Write minimal implementation**

Append to `eb_jepa/hierarchical.py`:

```python
@torch.no_grad()
def beam_candidates(jepa, z0, score_fn, horizon, beam_width, cell_size):
    """Guided beam over the frozen WM. Expand each surviving latent into its 4
    cardinal successors (one predictor step), keep the `beam_width` lowest
    `score_fn` (= distance-to-goal). Returns every kept node across depths 1..H:
      latents [M,D,1,1,1], first_dir [M], depth [M]. These are the reach-bounded
      subgoal CANDIDATES; `select_subgoal` picks among them.
    z0: [1,D,1,1,1]. score_fn(z[N,D,1,1,1]) -> [N], lower = closer to goal."""
    device = z0.device
    dirs = CARDINALS.to(device)                                  # [4,2]
    a0 = (dirs * cell_size).unsqueeze(-1)                        # [4,2,1]
    z = jepa.predictor(z0.expand(4, -1, -1, -1, -1).contiguous(), a0)  # [4,D,1,1,1]
    first = torch.arange(4, device=device)
    keep = min(beam_width, z.shape[0])
    sel = torch.topk(-score_fn(z), keep).indices
    beam_z, beam_first = z[sel], first[sel]
    all_z, all_first, all_depth = [beam_z], [beam_first], [torch.ones(keep, dtype=torch.long, device=device)]
    for h in range(2, horizon + 1):
        M = beam_z.shape[0]
        z_rep = beam_z.repeat_interleave(4, dim=0).contiguous()
        a = (dirs.repeat(M, 1) * cell_size).unsqueeze(-1)
        z_next = jepa.predictor(z_rep, a)                       # [4M,D,1,1,1]
        first_rep = beam_first.repeat_interleave(4)
        keep = min(beam_width, z_next.shape[0])
        sel = torch.topk(-score_fn(z_next), keep).indices
        beam_z, beam_first = z_next[sel], first_rep[sel]
        all_z.append(beam_z)
        all_first.append(beam_first)
        all_depth.append(torch.full((keep,), h, dtype=torch.long, device=device))
    return torch.cat(all_z), torch.cat(all_first), torch.cat(all_depth)


def select_subgoal(scores, depth, d_min):
    """Index of the lowest-score candidate with depth >= d_min (the reach-bounded,
    most goal-advancing subgoal). Falls back to global argmin if none qualify."""
    mask = depth >= d_min
    if mask.any():
        masked = scores.clone()
        masked[~mask] = float("inf")
        return int(torch.argmin(masked))
    return int(torch.argmin(scores))


@torch.no_grad()
def pick_action(jepa, d_head, z_t, z_sg, cell_size):
    """Low-level 1-step energy descent: pick the cardinal whose 1-step WM rollout
    minimizes d(predicted_latent, z_sg). Blocked moves -> WM predicts 'stay' ->
    high d -> not chosen. Returns a direction index 0..3."""
    device = z_t.device
    dirs = CARDINALS.to(device)                                 # [4,2]
    a = (dirs * cell_size).unsqueeze(-1)                        # [4,2,1]
    z_next = jepa.predictor(z_t.expand(4, -1, -1, -1, -1).contiguous(), a)  # [4,D,1,1,1]
    d = d_head(z_next, z_sg.expand(4, -1, -1, -1, -1))         # [4]
    return int(torch.argmin(d))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_latent_subgoal_planner.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add eb_jepa/hierarchical.py tests/test_latent_subgoal_planner.py
git commit -m "feat(planner): guided beam candidates, reach-bounded subgoal selection, 1-step descent"
```

---

## Task 4: Distance training script (cluster)

**Files:**
- Create: `examples/ac_video_jepa/maze/train_distance.py`

Integration script — the pure pieces it composes (`sample_pairs`, `distance_training_loss`,
`MRNDistanceHead`) are already unit-tested. Mirrors `main_subgoal.py`'s setup (frozen WM via
`build_fine`, the existing data pipeline with the warm-up + bf16 fixes).

- [ ] **Step 1: Write the script**

```python
# examples/ac_video_jepa/maze/train_distance.py
"""Train the latent quasimetric distance head d(z_a, z_b) on the FROZEN fine WM.

Self-supervised: encode full-length maze trajectories, regress d(z_i, z_j) -> (j-i).
A* is only the source of the trajectory data (shortest paths => exact step labels),
never a planner. WM is frozen. Output: <out_dir>/distance.pth.

Run: python -m examples.ac_video_jepa.maze.train_distance <fine_ckpt> <out_dir> [epochs=8]
"""
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW

from eb_jepa.datasets.utils import init_data
from eb_jepa.distance import sample_pairs, distance_training_loss
from eb_jepa.state_decoder import MRNDistanceHead
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def main():
    fine_ckpt, out_dir = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    pairs_per_batch = 256
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    cfg.data.sample_length = int(cfg.data.get("n_steps", 91)) - 1   # FULL path: full-range pairs
    cfg.data.batch_size = 128

    loader, _, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device)
    if data_pipeline is not None:
        data_pipeline.warm_up()

    jepa, f = build_fine(cfg, data_config, device)
    from eb_jepa.training_utils import load_checkpoint
    load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None,
                    device=device, strict=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    d_head = MRNDistanceHead(input_shape=f).to(device)
    opt = AdamW(d_head.parameters(), lr=1e-3, weight_decay=1e-5)
    print(f"[distance] f={f} epochs={epochs} pairs/batch={pairs_per_batch}", flush=True)

    for epoch in range(epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        for x, a, loc, _, _ in loader:
            x = x.to(device, non_blocking=True).float()        # bf16 -> fp32 (frozen encoder is fp32)
            with torch.no_grad():
                z = jepa.encode(x)                             # [B, f, T, 1, 1]
            T = z.shape[2]
            i, j = sample_pairs(T, pairs_per_batch)
            loss = distance_training_loss(d_head, z, i, j)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"[distance] epoch {epoch} {time.time()-t0:.0f}s loss={tot/max(nb,1):.4f}", flush=True)
        torch.save({"distance": d_head.state_dict(), "f": f},
                   os.path.join(out_dir, "distance.pth"))

    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[distance] DONE -> {out_dir}/distance.pth", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check locally**

Run: `uv run python -c "import ast; ast.parse(open('examples/ac_video_jepa/maze/train_distance.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add examples/ac_video_jepa/maze/train_distance.py
git commit -m "feat(distance): training script for the latent quasimetric head (frozen WM)"
```

---

## Task 5: Validation gate script (cluster)

**Files:**
- Create: `examples/ac_video_jepa/maze/validate_distance.py`

The §8 hard gate: on held-out mazes, `d(z_t, z_goal)` must decrease along the A\* path.
If `mean monotonic_fraction` is low, the distance is no good and the planner cannot work.

- [ ] **Step 1: Write the script**

```python
# examples/ac_video_jepa/maze/validate_distance.py
"""Validation gate for the distance head: walk the A* path of held-out mazes,
encode each state, and check d(z_t, z_goal) decreases monotonically toward the goal.

Run: python -m examples.ac_video_jepa.maze.validate_distance <fine_ckpt> <distance_pth> [num_ep=32]
Prints mean monotonic fraction. >= ~0.8 means the metric is usable.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.distance import monotonic_fraction
from eb_jepa.hierarchical import CARDINALS
from eb_jepa.state_decoder import MRNDistanceHead
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def main():
    fine_ckpt, dist_pth = sys.argv[1], sys.argv[2]
    num_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
                                    device=device)
    jepa, f = build_fine(cfg, env_config, device)
    load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None, device=device, strict=False)
    jepa.eval()
    ck = torch.load(dist_pth, map_location=device, weights_only=False)
    d_head = MRNDistanceHead(input_shape=ck["f"]).to(device)
    d_head.load_state_dict(ck["distance"]); d_head.eval()

    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=800, max_step_norm=1.5)
    norm = env.normalizer

    def enc(o):
        ot = norm.normalize_state(o.to(torch.float32, device=device)).unsqueeze(0).unsqueeze(2)
        return jepa.encode(ot)

    # map a (drow, dcol) cell delta to a CARDINALS index ([down,up,right,left] = [[1,0],[-1,0],[0,1],[0,-1]])
    DELTA = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}
    cell_size = float(env_config.cell_size)

    fracs = []
    for ep in range(num_ep):
        env.reset()
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
        z_goal = enc(info["target_obs"])
        # Walk the A* path by STEPPING THE ENV along it, encoding each obs, and recording
        # d(z_t, z_goal). A* only chooses which cells to visit for this metric check; the
        # planner at eval never sees it. (Same solve_a_star call eval_subgoal.py uses.)
        grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                              tuple(int(c) for c in env.goal_cell))
        if not solved:
            continue
        cells = solved[0]                                   # list of (row, col) along the path
        ds = [float(d_head(enc(obs), z_goal)[0])]
        for prev, nxt in zip(cells[:-1], cells[1:]):
            delta = (nxt[0] - prev[0], nxt[1] - prev[1])
            idx = DELTA.get(delta)
            if idx is None:
                break
            obs, _, _, _, info = env.step((CARDINALS[idx] * cell_size).cpu().numpy())
            ds.append(float(d_head(enc(obs), z_goal)[0]))
        fracs.append(monotonic_fraction(torch.tensor(ds)))
    print(f"[distance-validate] mean monotonic fraction = {np.mean(fracs):.3f} over {len(fracs)} mazes", flush=True)
    print("[distance-validate] GATE:", "PASS" if np.mean(fracs) >= 0.8 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm `solve_a_star`'s return shape**

The walk above assumes `solve_a_star(grid, start, goal)` returns `(cells, ...)` where
`cells` is a list of `(row, col)` tuples start->goal (matches `eval_subgoal.py`, which uses
`len(solved[0]) - 1` as the path length).

Run: `grep -nE "def solve_a_star|return" eb_jepa/datasets/maze/maze_solver.py | head`
Expected: returns a tuple whose first element is the path cell list. If the order is
goal->start, reverse `cells`. If consecutive cells can differ by more than one step,
the `DELTA.get(...) is None -> break` guard simply stops that maze early (safe).

- [ ] **Step 3: Syntax-check locally**

Run: `uv run python -c "import ast; ast.parse(open('examples/ac_video_jepa/maze/validate_distance.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add examples/ac_video_jepa/maze/validate_distance.py
git commit -m "feat(distance): validation gate (monotonicity of d along the A* path)"
```

---

## Task 6: A\*-free eval with the latent-subgoal planner (cluster)

**Files:**
- Create: `examples/ac_video_jepa/maze/eval_latent_subgoal.py`

Clone the structure of `eval_subgoal.py` (env setup, budget = `factor·len(A*)+margin`, SPL,
GIF dump, success via env `done`) but replace the high level + reacher with the new planner,
add a `seed` arg for reproducibility, and build `z_goal` from `info["target_obs"]`.

- [ ] **Step 1: Write the script**

```python
# examples/ac_video_jepa/maze/eval_latent_subgoal.py
"""A*-FREE maze eval with a LEARNED latent quasimetric (no SubgoalPredictor, no A*).

High level (every m steps): guided beam over the frozen WM -> candidates; subgoal =
argmin d(candidate, z_goal) with depth >= d_min. Low level (each step): pick the
cardinal minimizing d(1-step rollout, z_sg). Seeded + reports SPL.

Run: python -m examples.ac_video_jepa.maze.eval_latent_subgoal \
        <fine_ckpt> <distance_pth> <out_dir> <num_ep> <H> <d_min> <m> <beam_W> <budget_factor> <margin> <seed>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.datasets.maze.maze_solver import solve_a_star
from eb_jepa.hierarchical import beam_candidates, select_subgoal, pick_action, CARDINALS
from eb_jepa.state_decoder import MRNDistanceHead
from eb_jepa.training_utils import load_checkpoint
from examples.ac_video_jepa.maze.maze_fine_wm import build_fine


def main():
    a = sys.argv
    fine_ckpt, dist_pth, rdir = a[1], a[2], a[3]
    num_ep = int(a[4]); H = int(a[5]); d_min = int(a[6]); m = int(a[7]); beam_W = int(a[8])
    budget_factor = float(a[9]); margin = int(a[10]); seed = int(a[11]) if len(a) > 11 else 0
    os.makedirs(rdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
                                    device=device)
    cell_size = float(env_config.cell_size)

    jepa, f = build_fine(cfg, env_config, device)
    load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None, device=device, strict=False)
    jepa.eval()
    ck = torch.load(dist_pth, map_location=device, weights_only=False)
    d_head = MRNDistanceHead(input_shape=ck["f"]).to(device)
    d_head.load_state_dict(ck["distance"]); d_head.eval()

    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=800,
                     max_step_norm=1.5, rng=np.random.default_rng(seed))   # SEEDED
    norm = env.normalizer

    def enc(o):
        ot = norm.normalize_state(o.to(torch.float32, device=device)).unsqueeze(0).unsqueeze(2)
        return jepa.encode(ot)

    successes, spls = 0, []
    print(f"[latent-eval] H={H} d_min={d_min} m={m} W={beam_W} seed={seed} | {num_ep} mazes", flush=True)
    for ep in range(num_ep):
        obs, info = env.reset()
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
        z_goal = enc(info["target_obs"])
        # A* length sizes the budget only (never guides the agent) -- same as eval_subgoal.py
        grid = env.maze_grid.detach().cpu().numpy().astype(np.uint8)
        solved = solve_a_star(grid, tuple(int(c) for c in env.agent_cell),
                              tuple(int(c) for c in env.goal_cell))
        astar_len = (len(solved[0]) - 1) if solved else 100
        budget = min(int(budget_factor * astar_len + margin), 800)
        z_sg = None
        moves = 0; done = False
        for step in range(budget):
            z_t = enc(obs)
            if step % m == 0 or z_sg is None:
                score = lambda z: d_head(z, z_goal.expand(z.shape[0], -1, -1, -1, -1))
                zc, _first, depth = beam_candidates(jepa, z_t, score, H, beam_W, cell_size)
                idx = select_subgoal(score(zc), depth, d_min)
                z_sg = zc[idx:idx + 1]
            dir_idx = pick_action(jepa, d_head, z_t, z_sg, cell_size)
            obs, _, done, trunc, info = env.step((CARDINALS[dir_idx] * cell_size).cpu().numpy())
            moves += 1
            if done:
                break
        if done:
            successes += 1
            spls.append(astar_len / max(moves, astar_len))
        else:
            spls.append(0.0)
        print(f"[latent-eval] ep {ep}: {'SUCCESS' if done else 'fail'}", flush=True)

    sr = successes / num_ep
    spl = float(np.mean(spls))
    json.dump({"success_rate": sr, "spl": spl, "num_episodes": num_ep,
               "H": H, "d_min": d_min, "m": m, "beam_W": beam_W, "seed": seed, "astar_free": True},
              open(os.path.join(rdir, "latent_subgoal_eval.json"), "w"), indent=2)
    print(f"[latent-eval] A*-FREE success={sr*100:.2f}%  SPL={spl:.3f}  over {num_ep} mazes (seed {seed})", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm `create_env` forwards `rng` to the maze env**

The budget/goal/success expressions are copied verbatim from `eval_subgoal.py` (`solve_a_star`,
`env.maze_grid/agent_cell/goal_cell`, `info["target_obs"]`, success via `done`), so only the
seed plumbing is new. Verify the env accepts `rng`:

Run: `grep -nE "def __init__|rng|def create_env" eb_jepa/datasets/maze/env.py eb_jepa/datasets/utils.py | head`
Expected: `MazeEnv.__init__(..., rng=...)` exists and `create_env(env_name, config, **kwargs)` forwards kwargs. If `create_env` does NOT pass `**kwargs` to `MazeEnv`, add `rng=...` to the `MazeEnv` branch. (We confirmed `MazeEnv.__init__` has `rng: Optional[np.random.Generator] = None`.)

- [ ] **Step 3: Syntax-check locally**

Run: `uv run python -c "import ast; ast.parse(open('examples/ac_video_jepa/maze/eval_latent_subgoal.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add examples/ac_video_jepa/maze/eval_latent_subgoal.py
git commit -m "feat(eval): A*-free latent-subgoal eval (seeded, SPL) on the learned quasimetric"
```

---

## Task 7: sbatch pipeline + run on Dalia

**Files:**
- Create: `rerun_latent_subgoal.sh`

- [ ] **Step 1: Write the sbatch script**

```bash
# rerun_latent_subgoal.sh
#!/bin/bash
# Train the latent quasimetric distance head on the frozen aux WM, validate the
# monotonicity gate, then run the A*-free latent-subgoal eval. WM is NOT retrained.
#SBATCH --job-name=maze_latent_sg
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=maze_latent_sg_%j.out
#SBATCH --error=maze_latent_sg_%j.err
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
module load python312
uv sync --project "$REPO"

ROOT="${EBJEPA_CKPTS:-$EBJEPA_WORK/ckpts}/maze/baseline_full"
ALT="$EBJEPA_WORK/ckpts/maze/baseline_full"
if [ ! -f "$ROOT/aux/latest.pth.tar" ] && [ -f "$ALT/aux/latest.pth.tar" ]; then ROOT="$ALT"; fi
FINE="$ROOT/aux/latest.pth.tar"
DIST="$ROOT/distance"; mkdir -p "$DIST"
EVAL="$ROOT/eval_latent"; mkdir -p "$EVAL"
if [ ! -f "$FINE" ]; then echo "ERROR: aux ckpt not found at $FINE" >&2; exit 1; fi

echo ">>> [1/3] train distance head -> $DIST"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.train_distance "$FINE" "$DIST" 8

echo ">>> [2/3] validate distance gate"
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.validate_distance "$FINE" "$DIST/distance.pth" 32

echo ">>> [3/3] A*-free latent-subgoal eval -> $EVAL"
# num_ep H d_min m beam_W budget_factor margin seed
uv run --project "$REPO" python -m examples.ac_video_jepa.maze.eval_latent_subgoal \
    "$FINE" "$DIST/distance.pth" "$EVAL" 200 12 3 4 8 4 10 0

echo "=== DONE. success%/SPL above + $EVAL/latent_subgoal_eval.json ==="
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x rerun_latent_subgoal.sh
git add rerun_latent_subgoal.sh
git commit -m "feat(cluster): sbatch chain train->validate->eval for latent-subgoal planner"
```

- [ ] **Step 3: Push and run on Dalia**

```bash
git push origin main
# On Dalia (or have Claude do it over SSH):
#   cd /lustre/work/vivatech-equipe7/tfaure/eb_jepa && git pull origin main && sbatch rerun_latent_subgoal.sh
```

- [ ] **Step 4: Read the result**

Watch `maze_latent_sg_<job>.out` for:
1. `[distance] epoch N ... loss=...` decreasing — head is learning.
2. `[distance-validate] GATE: PASS` — **if FAIL, stop**: the metric is unusable, revisit the head/data (do not trust the eval).
3. `[latent-eval] A*-FREE success=..%  SPL=..` — the headline, vs the 66% baseline.

---

## Optional follow-up (not in MVP): swap MRN -> IQE

If long-range monotonicity (Task 5 gate) is weak, swap the head to IQE:
- `uv add torchqmet`
- replace `MRNDistanceHead` construction with `torchqmet.IQE(...)` wrapped to the same
  `(z_a, z_b) -> [N]` pooled interface; retrain (Task 4) and re-validate (Task 5). No other
  code changes — the planner and eval consume `d_head(z_a, z_b)` abstractly.
