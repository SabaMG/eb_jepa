"""Maze normalizer — same interface as ``two_rooms/normalizer.py``."""

import torch


class MazeNormalizer:
    """Normalises pixel-space states (2 channels) and locations.

    Stats are computed analytically from the maze geometry rather than from
    data: locations are uniformly distributed in [0, img_size); the agent
    channel is a Gaussian dot covering ~constant area per frame; the maze
    channel is a static binary mask.
    """

    def __init__(self, img_size: int = 63):
        self.img_size = img_size
        # Uniform-on-[0, img_size) → mean (img_size-1)/2, std img_size/sqrt(12)
        mid = (img_size - 1) / 2.0
        std = img_size / (12.0 ** 0.5)
        self.location_mean = torch.tensor([mid, mid])
        self.location_std = torch.tensor([std, std])

        # Conservative defaults; states are min-max normalised first so the
        # absolute scale here is not critical.
        self.state_mean = torch.tensor([0.05, 0.5])
        self.state_std = torch.tensor([0.10, 0.5])

    def min_max_normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) >= 3:
            state = state - state.amin(dim=(-2, -1), keepdim=True)
            state = state / (state.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        else:
            state = state - state.amin(dim=-1, keepdim=True)
            state = state / (state.amax(dim=-1, keepdim=True) + 1e-6)
        return state

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state = self.min_max_normalize_state(state)
        mean = self.state_mean.view(-1, 1, 1).to(state.device)
        std = self.state_std.view(-1, 1, 1).to(state.device) + 1e-6
        ch = state.shape[-3]
        if ch < mean.shape[0] and not (mean.shape[0] % ch):
            mean = mean[:ch]
            std = std[:ch]
        return (state - mean) / std

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = self.state_mean.view(-1, 1, 1).to(state.device)
        std = self.state_std.view(-1, 1, 1).to(state.device)
        ch = state.shape[-3]
        if ch < mean.shape[0] and not (mean.shape[0] % ch):
            mean = mean[:ch]
            std = std[:ch]
        return state * std + mean

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return (location - self.location_mean.to(location.device)) / (
            self.location_std.to(location.device) + 1e-6
        )

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        return location * self.location_std.to(location.device) + self.location_mean.to(
            location.device
        )

    def unnormalize_mse(self, mse):
        return mse * (self.location_std.mean().to(mse.device) ** 2)
