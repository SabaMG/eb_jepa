import torch
from torch import nn


class GoalValueHead(nn.Module):
    """Goal-conditioned scalar value V(z_state, z_goal) — TD-MPC style.

    Replaces the crude "distance-in-latent-space" planning cost with a *learned*
    value (Hansen et al., TD-MPC 2022/2024): the head maps a state latent and a
    goal latent to a scalar in (0, 1) interpreted as the discounted return-to-goal
    (≈ ``gamma ** steps_to_goal``). Trained by TD on the world model's own
    rollouts (see ``examples/ac_video_jepa/main.py``); at planning time the MPC
    objective MAXIMISES this value, so the planner optimises a quantity that
    correlates with task success rather than raw representation distance.

    Mirrors ``MLPXYHead``'s pooled-latent interface: latents are [B, C, T, h, w]
    with h=w=1 for the impala encoder, so spatial mean-pooling is a no-op there.
    """

    def __init__(self, input_shape, hidden=512):  # input_shape = C (channel dim)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * input_shape, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, goal):
        """
        Args:
            state: [B, C, T, h, w]
            goal:  [B or 1, C, 1, h, w]
        Returns:
            value: [B, T] in (0, 1)
        """
        bs, c, t, h, w = state.shape
        s = state.mean(dim=(3, 4)).permute(0, 2, 1)        # [B, T, C]
        g = goal.mean(dim=(3, 4)).permute(0, 2, 1)         # [B or 1, 1, C]
        g = g.expand(bs, t, c)                             # [B, T, C]
        feat = torch.cat([s, g], dim=-1)                   # [B, T, 2C]
        v = self.mlp(feat).squeeze(-1)                     # [B, T]
        return torch.sigmoid(v)


class MLPXYHead(nn.Module):
    """A head to recover the xy location from features."""

    def __init__(self, input_shape, normalizer=None):  # input_shape = (C, H, W)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_shape, 512), nn.ReLU(inplace=True), nn.Linear(512, 2)
        )
        self.normalizer = normalizer

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            pred: [B, 2, T]
        """
        bs, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(bs * t, c, h, w)  # [B*T, C, H, W]

        x = x.squeeze(-1).squeeze(-1)  # [B*T, C]

        pred = self.mlp(x)

        pred = pred.view(bs, t, 2).permute(0, 2, 1)

        return pred
