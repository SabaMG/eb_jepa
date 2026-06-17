"""Hierarchical (two-level) maze navigation primitives — A*-FREE.

The maze impala encoder pools to a 1x1 latent, so a state is a vector z in R^D.
Two levels (trained WITH A*, evaluated WITHOUT any A*):

- HIGH level: ``SubgoalPredictor(z_current, goal_xy) -> next waypoint position``
  (feudal/subgoal style; learned replacement for A* waypoint generation).
- LOW level: reach that waypoint with the frozen, wall-aware fine world model.
  ``fine_kstep_target`` rolls the fine WM K steps in a cardinal direction and
  returns the resulting latent — the K-step lookahead used by the reacher.

See ``examples/ac_video_jepa/main_subgoal.py`` (train the high level) and
``eval_subgoal.py`` (A*-free closed-loop eval). Co-training the two levels is in
``main_cotrain.py``.
"""
import torch
import torch.nn as nn

# 4 cardinal directions as unit (row, col) steps; scaled by cell_size at use.
CARDINALS = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


class SubgoalPredictor(nn.Module):
    """High-level policy that REPLACES A* waypoint generation (feudal/subgoal style).

    Given the current state latent (which encodes the WHOLE maze — the wall mask is
    in the obs image — plus the agent position) and the goal position, predict the
    position of the NEXT waypoint ~N cells along the route to the goal. Trained
    SUPERVISED on A* trajectories (label = the A* position N steps ahead), so at
    eval it proposes waypoints itself and the low-level reacher follows them — A*
    is used only as a training teacher, never at eval.
    """

    def __init__(self, dim, hidden=512):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z, goal_xy):
        """z: [B,dim,1,1,1] (or [B,dim]); goal_xy: [B,2] (normalized). -> [B,2]."""
        v = z.reshape(z.shape[0], self.dim)
        return self.net(torch.cat([v, goal_xy], dim=-1))


@torch.no_grad()
def fine_kstep_target(jepa, obs_init, dir_idx, K, cell_size, ctxt_window_time=1):
    """Roll the frozen fine world model K steps with a CONSTANT cardinal action and
    return the resulting latent. The fine WM is wall-aware (it predicts "stay" into
    a wall), so this K-step lookahead lets the low-level reacher score each direction
    by how close its K-step endpoint lands to the waypoint, avoiding dead-ends.
    obs_init: [B,C,1,H,W]; dir_idx: [B] long. Returns [B,D,1,1,1]."""
    dirs = CARDINALS.to(obs_init.device)[dir_idx]          # [B,2]
    a = (dirs * cell_size).unsqueeze(-1).repeat(1, 1, K)   # [B,2,K]
    pred, _ = jepa.unroll(obs_init, a, nsteps=K, unroll_mode="autoregressive",
                          ctxt_window_time=ctxt_window_time, compute_loss=False,
                          return_all_steps=False)           # [B,D,1+K,1,1]
    return pred[:, :, -1:]                                   # [B,D,1,1,1]
