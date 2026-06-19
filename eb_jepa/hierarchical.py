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


@torch.no_grad()
def fine_beam_dist(jepa, xy_head, obs_init, waypoint, depth, beam_width, cell_size):
    """BEAM SEARCH reacher over the frozen WM (the improvement over the greedy
    constant-direction K-step lookahead in ``fine_kstep_target``).

    The greedy reacher only scores 4 *straight* K-step rolls, so it cannot see a
    route that needs a turn within the horizon (e.g. right-then-down around a wall
    stub). Here we roll the frozen, wall-aware predictor forward as a real search
    over SEQUENCES of cardinals: at each depth every surviving beam is expanded
    into its 4 cardinal successors (one GRU step in latent space), scored by the
    probe-decoded distance to the waypoint, and the top ``beam_width`` are kept.

    Drop-in for the eval reacher: returns a list ``dist[4]`` = the best (smallest)
    distance-to-waypoint reachable by a route whose FIRST move is direction d
    (D=down, U=up, R=right, L=left), exactly the per-direction score the existing
    selection loop (blocked-skip + no-U-turn) consumes. So swapping greedy->beam
    is a controlled, low-level-only change.

    A blocked first move is handled for free: the wall-aware WM predicts "stay",
    so that branch's latent doesn't advance toward the waypoint -> high dist ->
    deprioritised (and the env-level blocked-skip still catches a real bump).

    obs_init: [1,C,1,H,W]; waypoint: [2] (normalized). depth>=1.
    """
    device = obs_init.device
    dirs = CARDINALS.to(device)                              # [4,2]
    z0 = jepa.encode(obs_init)                               # [1,D,1,1,1]

    def probe_dist(z):                                       # z:[N,D,1,1,1] -> [N]
        xy = xy_head(z.float()).permute(0, 2, 1)[:, 0]      # [N,2] normalized
        return torch.norm(xy - waypoint.unsqueeze(0), dim=1)

    # Depth 0: expand the root into its 4 cardinal successors (the first move).
    a0 = (dirs * cell_size).unsqueeze(-1)                    # [4,2,1]
    z1 = jepa.predictor(z0.expand(4, -1, -1, -1, -1).contiguous(), a0)  # [4,D,1,1,1]
    d1 = probe_dist(z1)                                      # [4]
    out = d1.clone()                                         # per-first-dir best dist
    beam_z = z1                                              # [M,D,1,1,1]
    beam_first = torch.arange(4, device=device)             # [M] first action id
    beam_best = d1.clone()                                  # [M] min dist along beam

    for _ in range(depth - 1):
        M = beam_z.shape[0]
        z_rep = beam_z.repeat_interleave(4, dim=0).contiguous()         # [4M,...]
        a = (dirs.repeat(M, 1) * cell_size).unsqueeze(-1)              # [4M,2,1]
        z_next = jepa.predictor(z_rep, a)                              # [4M,D,1,1,1]
        first_rep = beam_first.repeat_interleave(4)                    # [4M]
        d = probe_dist(z_next)                                         # [4M]
        best_rep = torch.minimum(beam_best.repeat_interleave(4), d)    # [4M]
        # fold each child's best-so-far into its first-direction score
        out.scatter_reduce_(0, first_rep, best_rep, reduce="amin")
        k = min(beam_width, d.shape[0])
        sel = torch.topk(-best_rep, k).indices
        beam_z, beam_first, beam_best = z_next[sel], first_rep[sel], best_rep[sel]

    return [float(x) for x in out]
