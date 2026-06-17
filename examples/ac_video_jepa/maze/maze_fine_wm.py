"""Shared helper: rebuild the FINE maze world model (encoder + RNN predictor) so a
checkpoint can be loaded for frozen-world-model use (A*-free planning, probing).

The maze fine WM is a stride-1 ImpalaEncoder (1x1 latent) + RNNPredictor, trained
on A* trajectories. Loading it for inference/planning only needs the encoder +
predictor; the regularizer is replaced by a no-op so ``JEPA.unroll(compute_loss=
False)`` works without rebuilding the full training objective. Used by
``main_subgoal.py`` (train the high level) and ``eval_subgoal.py`` (A*-free eval).
"""
import torch
import torch.nn as nn

from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq


class _DummyReg(nn.Module):
    """No-op regularizer: only ``unroll(compute_loss=False)`` is used at inference."""

    def forward(self, state, actions):
        z = torch.tensor(0.0, device=state.device)
        return z, z, {}


def build_fine(cfg, data_config, device):
    """Build the fine maze world model (encoder + RNN predictor). Returns (jepa, f)
    where f is the latent channel dim. Load weights with strict=False afterwards."""
    enc = ImpalaEncoder(width=1, stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
                        num_blocks=2, dropout_rate=None, layer_norm=False,
                        input_channels=cfg.model.dobs, final_ln=True, mlp_output_dim=512,
                        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size))
    f = enc(torch.rand(1, cfg.model.dobs, 1, data_config.img_size, data_config.img_size)).shape[1]
    pred = RNNPredictor(hidden_size=enc.mlp_output_dim, final_ln=enc.final_ln)
    jepa = JEPA(enc, nn.Identity(), pred, _DummyReg(), SquareLossSeq()).to(device)
    return jepa, f
