"""Strict 2-level Hierarchical JEPA (H-JEPA) for the maze — TWO abstractions.

Level 0 (fine, FROZEN): the existing world model. Predicts the fine latent z one
cell-step ahead (`jepa.predictor`), and acts as the wall-aware simulator for the
coarse targets (`fine_kstep_target`).

Level 1 (coarse, LEARNED): a second abstraction.
  - CoarseEncoder  psi:  z (pooled fine latent) -> s   (coarse state)
  - CoarsePredictor P_high: (s, option o in 0..3) -> s_hat  (coarse state after a
    k-step MACRO-option = "commit to cardinal o for k fine steps")

The coarse level is trained as a JEPA: predict psi_ema(z_{t+k}) from (psi(z_t), o),
with VICReg (std + covariance) preventing collapse. At plan time the coarse level
searches over macro-options (long horizon, cheap), and the fine level executes the
chosen coarse subgoal one cell-step at a time. See the design spec
docs/superpowers/specs/2026-06-20-hjepa-two-level-design.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.hierarchical import CARDINALS

N_OPTIONS = 4  # the 4 macro-cardinals (down, up, right, left), held k fine steps


@torch.no_grad()
def dream_macro_option(jepa, z0, option, k, cell_size, stop_eps=1e-3):
    """DREAM a directional macro-option ENTIRELY in latent space (LeCun-style WM
    imagination): roll the frozen world-model PREDICTOR k steps under cardinal
    `option`, never decoding back to pixels. Early-stops per-sample when the latent
    stops moving (a wall -> the wall-aware WM predicts "stay"), so the dream lands at
    the wall instead of pushing through it.

    This replaces `fine_kstep_target` (which re-encoded through `jepa.unroll` and
    always rolled a fixed straight line): the dream stays in representation space and
    stops at walls. It generates the coarse-level training targets and is the
    semantics of a "commit to direction o" macro-option.

    z0: [N, D, 1, 1, 1]; option: int or long[N]. Returns z_end: [N, D, 1, 1, 1].
    """
    device = z0.device
    dirs = CARDINALS.to(device)
    if not torch.is_tensor(option):
        option = torch.full((z0.shape[0],), int(option), device=device, dtype=torch.long)
    a = (dirs[option] * cell_size).unsqueeze(-1)            # [N, 2, 1]
    z = z0
    for _ in range(k):
        z_next = jepa.predictor(z, a)                      # [N, D, 1, 1, 1]
        moved = (z_next - z).flatten(1).norm(dim=1) > stop_eps          # [N]
        z = torch.where(moved.view(-1, 1, 1, 1, 1), z_next, z)         # freeze stalled samples
        if not moved.any():
            break
    return z


class CoarseEncoder(nn.Module):
    """Level-1 abstraction psi: fine latent z -> coarse state s.

    z is a pooled-latent [N, C, *spatial] (e.g. [N, 512, 1, 1, 1] from the impala
    encoder); dims >= 2 are mean-pooled to [N, C], then an MLP maps to [N, coarse_dim].
    """

    def __init__(self, in_dim, coarse_dim=128, hidden=512):
        super().__init__()
        self.coarse_dim = coarse_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, coarse_dim),
        )

    @staticmethod
    def _pool(z):
        if z.dim() <= 2:
            return z
        return z.flatten(2).mean(dim=-1)

    def forward(self, z):
        return self.net(self._pool(z))


class CoarsePredictor(nn.Module):
    """Level-1 coarse JEPA predictor: (s, option) -> next coarse state s_hat.

    `option` is an int in [0, n_options) or a long tensor [N]; it is one-hot encoded
    and concatenated to the coarse state.
    """

    def __init__(self, coarse_dim=128, n_options=N_OPTIONS, hidden=512):
        super().__init__()
        self.n_options = n_options
        self.net = nn.Sequential(
            nn.Linear(coarse_dim + n_options, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, coarse_dim),
        )

    def forward(self, s, o):
        if not torch.is_tensor(o):
            o = torch.full((s.shape[0],), int(o), device=s.device, dtype=torch.long)
        oh = F.one_hot(o, self.n_options).to(s.dtype)
        return self.net(torch.cat([s, oh], dim=-1))


@torch.no_grad()
def ema_update(target, online, tau=0.99):
    """In-place EMA: target <- tau*target + (1-tau)*online (BYOL/JEPA target net)."""
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(tau).add_(po, alpha=1.0 - tau)
    for bt, bo in zip(target.buffers(), online.buffers()):
        bt.copy_(bo)


def coarse_jepa_loss(psi, p_high, s_targets, z_t, std_loss_fn, cov_loss_fn,
                     std_coeff=16.0, cov_coeff=8.0):
    """One coarse-JEPA training loss.

    Args:
        psi, p_high: the level-1 modules (online).
        s_targets: [N, n_options, coarse_dim] target coarse states from psi_ema on the
            frozen-WM k-step rolls (already detached by the caller).
        z_t: [N, C, *spatial] fine latents at the start states.
        std_loss_fn, cov_loss_fn: VICReg HingeStdLoss / CovarianceLoss instances.
    Returns:
        (loss, pred_loss_detached, reg_detached)
    """
    s_t = psi(z_t)                                                  # [N, dc]
    pred = torch.stack([p_high(s_t, o) for o in range(p_high.n_options)], dim=1)  # [N, nopt, dc]
    pred_loss = F.mse_loss(pred, s_targets)
    reg = std_coeff * std_loss_fn(s_t) + cov_coeff * cov_loss_fn(s_t)
    loss = pred_loss + reg
    return loss, pred_loss.detach(), reg.detach()


@torch.no_grad()
def coarse_beam(p_high, s0, s_goal, horizon, width):
    """HIGH level: plan over macro-options in COARSE space with the coarse predictor.

    Roll P_high forward over option sequences to `horizon`, scoring each rolled coarse
    state by Euclidean distance to s_goal; keep the best `width`. Returns the first
    macro-option of the best plan and the coarse subgoal it predicts.

    s0, s_goal: [1, coarse_dim]. Returns (option_index:int, s_sg:[1, coarse_dim]).
    """
    device = s0.device
    nopt = p_high.n_options
    s = torch.cat([p_high(s0, o) for o in range(nopt)], dim=0)      # [nopt, dc]
    first = torch.arange(nopt, device=device)
    out_score = torch.norm(s - s_goal, dim=-1)                     # [nopt] best-so-far per first option
    keep = min(width, nopt)
    sel = torch.topk(-out_score, keep).indices
    beam_s, beam_first = s[sel], first[sel]
    for _ in range(horizon - 1):
        M = beam_s.shape[0]
        cand = torch.cat([p_high(beam_s, o) for o in range(nopt)], dim=0)   # [nopt*M, dc]
        first_rep = beam_first.repeat(nopt)                                  # matches dim-0 order
        score = torch.norm(cand - s_goal, dim=-1)                           # [nopt*M]
        out_score = out_score.scatter_reduce(0, first_rep, score, reduce="amin")
        keep = min(width, cand.shape[0])
        sel = torch.topk(-score, keep).indices
        beam_s, beam_first = cand[sel], first_rep[sel]
    o_star = int(torch.argmin(out_score))
    s_sg = p_high(s0, o_star)                                       # [1, dc]
    return o_star, s_sg


@torch.no_grad()
def rank_fine_actions(jepa, psi, z_t, s_sg, cell_size, depth=1, width=4, block_eps=1e-3):
    """Rank the 4 first-cardinals by how close a `depth`-step WM DREAM gets (in coarse
    space) to the subgoal s_sg, best first. Two upgrades, both training-free:

    - MODEL-BASED wall avoidance (dreamer-scale): a move whose dream doesn't change the
      latent (`||predictor(z,a) - z|| < block_eps`) is a wall -> pushed to the back. The
      agent avoids walls *in imagination*, no bumping.
    - `depth > 1`: a fine-level DREAM LOOKAHEAD (a small beam over cardinal sequences),
      so the agent routes around a local wall by imagining `depth` steps, not just 1.
      `depth=1` reduces to the 1-step rule. Returns a list of dir indices 0..3."""
    INF = 1e6
    device = z_t.device
    dirs = CARDINALS.to(device)
    a1 = (dirs * cell_size).unsqueeze(-1)                                  # [4,2,1]
    z1 = jepa.predictor(z_t.expand(4, -1, -1, -1, -1).contiguous(), a1)    # [4,D,1,1,1]
    moved = (z1 - z_t.expand_as(z1)).flatten(1).norm(dim=1) > block_eps
    first = torch.arange(4, device=device)
    d1 = torch.where(moved, torch.norm(psi(z1) - s_sg, dim=-1),
                     torch.full((4,), INF, device=device))
    out = d1.clone()                                                       # best dist per first action
    if depth > 1:
        keep = min(width, 4)
        sel = torch.topk(-d1, keep).indices
        beam_z, beam_first = z1[sel], first[sel]
        for _ in range(depth - 1):
            M = beam_z.shape[0]
            z_rep = beam_z.repeat_interleave(4, dim=0).contiguous()
            a = (dirs.repeat(M, 1) * cell_size).unsqueeze(-1)
            z_next = jepa.predictor(z_rep, a)
            mv = (z_next - z_rep).flatten(1).norm(dim=1) > block_eps
            first_rep = beam_first.repeat_interleave(4)
            dd = torch.where(mv, torch.norm(psi(z_next) - s_sg, dim=-1),
                             torch.full((z_next.shape[0],), INF, device=device))
            out = out.scatter_reduce(0, first_rep, dd, reduce="amin")
            keep = min(width, z_next.shape[0])
            sel = torch.topk(-dd, keep).indices
            beam_z, beam_first = z_next[sel], first_rep[sel]
    return torch.argsort(out).tolist()


@torch.no_grad()
def pick_fine_action(jepa, psi, z_t, s_sg, cell_size):
    """LOW level: pick the cardinal whose 1-step fine prediction lands closest (in
    COARSE space) to the coarse subgoal s_sg. Energy descent in s-space via the fine WM.

    z_t: [1, D, 1, 1, 1]; s_sg: [1, coarse_dim]. Returns a direction index 0..3.
    """
    device = z_t.device
    dirs = CARDINALS.to(device)                                    # [4, 2]
    a = (dirs * cell_size).unsqueeze(-1)                           # [4, 2, 1]
    z_next = jepa.predictor(z_t.expand(4, -1, -1, -1, -1).contiguous(), a)  # [4, D, 1, 1, 1]
    s_next = psi(z_next)                                           # [4, dc]
    d = torch.norm(s_next - s_sg, dim=-1)                          # [4]
    return int(torch.argmin(d))
