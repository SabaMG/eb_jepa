import torch

from eb_jepa.hjepa import (
    CoarseDistanceHead,
    CoarseEncoder,
    CoarsePredictor,
    coarse_beam,
    coarse_jepa_loss,
    dream_macro_option,
    ema_update,
    pick_fine_action,
)


def test_coarse_distance_head_quasimetric():
    torch.manual_seed(0)
    d = CoarseDistanceHead(coarse_dim=16, embed=16, asym=16)
    a, b, c = torch.randn(20, 16), torch.randn(20, 16), torch.randn(20, 16)
    assert (d(a, b) >= 0).all()                                   # non-negative
    assert torch.allclose(d(a, a), torch.zeros(20), atol=1e-5)    # identity = 0
    assert (d(a, c) <= d(a, b) + d(b, c) + 1e-4).all()           # triangle inequality


def test_coarse_encoder_layernorm_bounds_scale():
    psi = CoarseEncoder(in_dim=32, coarse_dim=16, layer_norm=True)
    s = psi(torch.randn(8, 32) * 50)        # large inputs
    assert s.std().item() < 3.0             # LayerNorm keeps the coarse scale bounded


def test_coarse_beam_accepts_dist_fn():
    p = CoarsePredictor(coarse_dim=8, n_options=4)
    s0, s_goal = torch.zeros(1, 8), torch.randn(1, 8)
    dist_fn = lambda sa, sb: torch.norm(sa - sb, dim=-1)   # Euclidean via dist_fn == default
    o1, _ = coarse_beam(p, s0, s_goal, horizon=2, width=4)
    o2, _ = coarse_beam(p, s0, s_goal, horizon=2, width=4, dist_fn=dist_fn)
    assert o1 == o2                                         # dist_fn=Euclidean matches default
from eb_jepa.losses import CovarianceLoss, HingeStdLoss


class _StubJEPA:
    """Latent = a 2D 'position' in channels [0,1]; predictor adds the action."""
    def predictor(self, z, a):
        out = z.clone()
        out[:, 0, 0, 0, 0] += a[:, 0, 0]
        out[:, 1, 0, 0, 0] += a[:, 1, 0]
        return out


def test_dream_macro_option_rolls_in_latent():
    jepa = _StubJEPA()
    z0 = torch.zeros(1, 4, 1, 1, 1)
    z_end = dream_macro_option(jepa, z0, option=0, k=3, cell_size=1.0)  # CARDINALS[0]=[1,0]
    assert torch.isclose(z_end[0, 0, 0, 0, 0], torch.tensor(3.0))       # moved +3 in x
    assert torch.isclose(z_end[0, 1, 0, 0, 0], torch.tensor(0.0))


def test_dream_macro_option_early_stops_when_blocked():
    jepa = _StubJEPA()
    z0 = torch.randn(2, 4, 1, 1, 1)
    z_end = dream_macro_option(jepa, z0, option=0, k=5, cell_size=0.0)  # zero action -> no move
    assert torch.allclose(z_end, z0)                                    # stalled immediately


# ---------- modules ----------

def test_coarse_encoder_shape_and_pooling():
    psi = CoarseEncoder(in_dim=32, coarse_dim=8)
    z = torch.randn(5, 32, 1, 1, 1)          # impala-style pooled latent
    s = psi(z)
    assert s.shape == (5, 8)
    z2 = torch.randn(5, 32)                  # already pooled
    assert psi(z2).shape == (5, 8)


def test_coarse_predictor_int_and_tensor_option():
    p = CoarsePredictor(coarse_dim=8, n_options=4)
    s = torch.randn(6, 8)
    assert p(s, 2).shape == (6, 8)                       # int option
    assert p(s, torch.tensor([0, 1, 2, 3, 0, 1])).shape == (6, 8)  # tensor option


def test_ema_update_moves_target_toward_online():
    online = CoarsePredictor(coarse_dim=8)
    target = CoarsePredictor(coarse_dim=8)
    target.load_state_dict(online.state_dict())
    with torch.no_grad():                                # perturb online
        for pp in online.parameters():
            pp.add_(1.0)
    ema_update(target, online, tau=0.0)                  # tau=0 -> target := online
    for pt, po in zip(target.parameters(), online.parameters()):
        assert torch.allclose(pt, po)


# ---------- coarse JEPA loss ----------

def test_coarse_jepa_loss_scalar_and_grad():
    torch.manual_seed(0)
    psi = CoarseEncoder(in_dim=16, coarse_dim=8)
    p = CoarsePredictor(coarse_dim=8, n_options=4)
    z_t = torch.randn(12, 16)
    s_targets = torch.randn(12, 4, 8)
    loss, pred, reg = coarse_jepa_loss(psi, p, s_targets, z_t,
                                       HingeStdLoss(), CovarianceLoss())
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(pp.grad is not None for pp in psi.parameters())
    assert any(pp.grad is not None for pp in p.parameters())


def test_coarse_jepa_loss_penalizes_collapse():
    # identical inputs -> identical coarse states -> zero std -> high HingeStd penalty
    psi = CoarseEncoder(in_dim=16, coarse_dim=8)
    p = CoarsePredictor(coarse_dim=8, n_options=4)
    z_same = torch.ones(12, 16)
    s_targets = torch.zeros(12, 4, 8)
    _, _, reg = coarse_jepa_loss(psi, p, s_targets, z_same,
                                 HingeStdLoss(), CovarianceLoss())
    assert reg > 0.5   # std hinge fires when features have ~zero variance


# ---------- planner ----------

class _StubPredictor:
    """p_high(s, o) = s + OFFSET[o]; lets us check the beam heads toward s_goal."""
    n_options = 4

    def __init__(self):
        self.offset = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    def __call__(self, s, o):
        if not torch.is_tensor(o):
            return s + self.offset[int(o)]
        return s + self.offset[o]


def test_coarse_beam_picks_option_toward_goal():
    p = _StubPredictor()
    s0 = torch.zeros(1, 2)
    s_goal = torch.tensor([[2.0, 0.0]])      # two steps in option 0's direction
    o_star, s_sg = coarse_beam(p, s0, s_goal, horizon=2, width=4)
    assert o_star == 0
    assert torch.allclose(s_sg, torch.tensor([[1.0, 0.0]]))   # one macro-step toward goal


def test_rank_fine_actions_deprioritizes_dreamed_walls():
    from eb_jepa.hjepa import rank_fine_actions

    class WallJEPA:
        # cardinal 0 = [1,0] is a 'wall': the WM predicts stay (no motion); others move
        def predictor(self, z, a):
            out = z.clone()
            for i in range(z.shape[0]):
                ax, ay = float(a[i, 0, 0]), float(a[i, 1, 0])
                if ax > 0:        # dir 0 -> blocked, predict 'stay'
                    continue
                out[i, 0, 0, 0, 0] += ax
                out[i, 1, 0, 0, 0] += ay
            return out

    psi = lambda z: z.flatten(2).mean(-1)[:, :2]
    z_t = torch.zeros(1, 4, 1, 1, 1)
    s_sg = torch.tensor([[1.0, 0.0]])    # subgoal points exactly where the BLOCKED dir 0 would go
    order = rank_fine_actions(WallJEPA(), psi, z_t, s_sg, cell_size=1.0)
    assert order[-1] == 0                # dir 0 is a dreamed wall -> ranked last despite pointing at goal


def test_dream_subgoal_targets_toward_goal():
    from eb_jepa.hjepa import dream_subgoal

    class J:
        def predictor(self, z, a):
            o = z.clone(); o[:, 0, 0, 0, 0] += a[:, 0, 0]; o[:, 1, 0, 0, 0] += a[:, 1, 0]; return o

    psi = lambda z: z.flatten(2).mean(-1)[:, :2]      # coarse state = position
    z_t = torch.zeros(1, 4, 1, 1, 1)
    s_goal = torch.tensor([[3.0, 0.0]])               # goal at +x
    s_sg, z_sg = dream_subgoal(J(), psi, z_t, s_goal, horizon=4, width=4, d_min=2, cell_size=1.0)
    pos = z_sg[0, :2, 0, 0, 0]
    assert torch.norm(pos - torch.tensor([3.0, 0.0])) < 3.0   # subgoal is closer to goal than the start


def test_pick_fine_action_descends_coarse_distance():
    jepa = _StubJEPA()
    psi = lambda z: z.flatten(2).mean(-1)[:, :2]   # coarse state = position
    z_t = torch.zeros(1, 4, 1, 1, 1)
    s_sg = torch.tensor([[1.0, 0.0]])              # subgoal: +1 in x (CARDINALS row 0 = [1,0])
    a = pick_fine_action(jepa, psi, z_t, s_sg, cell_size=1.0)
    assert a == 0
