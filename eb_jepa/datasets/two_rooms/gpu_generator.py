"""Vectorised, fully on-GPU port of ``WallDataset`` sample generation.

Pack 8 reuses this generator for *offline* preprocessing: it produces whole
batches of two-rooms samples directly on the GPU, which the preprocessing
script then writes to disk. It is a verbatim port of pack7's generator with the
streaming/double-buffer machinery removed (pack8 does not stream; it generates
once, up front, and saves to a memmap).

Fidelity: the generation is a *vectorised* port of the per-sample CPU generator
in ``wall_dataset.py`` + ``dot_dataset.py``. Same generative process and
collision physics, batched across samples. The RNG draw order differs from the
CPU code, so batches are not bit-identical, but the per-sample distribution is
identical (validated for pack7 at 95% planning success).

All randomness is kept sync-free: ``scipy`` truncnorm is replaced by an
inverse-CDF sampler (erfinv), and ``torch.distributions.VonMises`` is replaced
by a fixed-round vectorised Best-Fisher sampler.

The generator honours ``config.normalize``: with ``normalize=False`` it returns
the *raw* rendered states (uint8-valued floats in [0, 255]) and raw locations,
which is what the offline preprocessor stores (normalisation is then applied at
load time, exactly as the base ``WallDataset`` does).
"""

import math

import torch

from eb_jepa.datasets.two_rooms.normalizer import Normalizer
from eb_jepa.datasets.two_rooms.utils import generate_wall_layouts
from eb_jepa.datasets.two_rooms.wall_dataset import WallDatasetConfig

_SQRT2 = math.sqrt(2.0)


class GPUWallGenerator:
    """Vectorised, fully on-GPU port of ``WallDataset`` sample generation.

    Produces batches as a dict with keys ``states`` (B, 2, sample_length, H, W),
    ``actions`` (B, 2, sample_length), ``locations`` (B, 2, sample_length),
    ``wall_x`` (B,) and ``door_y`` (B,).
    """

    def __init__(self, config: WallDatasetConfig, device, gen_batch_size=None):
        # The GPU path reproduces exactly the trajectory mix used in training.
        # The rarely-used branches are not ported; fail loudly instead of
        # silently diverging from the CPU generator.
        if config.fix_wall_batch_k is not None:
            raise NotImplementedError(
                "GPUWallGenerator does not support fix_wall_batch_k; got "
                f"{config.fix_wall_batch_k}"
            )
        if config.wall_bump_rate > 0 or config.expert_cross_wall_rate > 0:
            raise NotImplementedError(
                "GPUWallGenerator supports cross_wall_rate only "
                "(wall_bump_rate and expert_cross_wall_rate must be 0)"
            )
        if config.dup_traj_rate > 0:
            raise NotImplementedError("GPUWallGenerator does not support dup_traj_rate")
        if config.n_steps_reduce_factor != 1:
            raise NotImplementedError(
                "GPUWallGenerator only supports n_steps_reduce_factor == 1"
            )

        self.config = config
        self.device = torch.device(device)
        self.gen_batch_size = gen_batch_size
        self.normalizer = Normalizer()

        # Pre-build the (wall_x, door_y) layout table. Sampling a layout code
        # uniformly == sampling a (wall_pos, door_pos) pair uniformly (the v/h
        # "type" is irrelevant to this generation path — both render the same
        # vertical wall), so we just index this table with a uniform integer.
        layouts, _ = generate_wall_layouts(config)
        codes = list(layouts.keys())
        self._wall_pos = torch.tensor(
            [layouts[c]["wall_pos"] for c in codes],
            dtype=torch.float32,
            device=self.device,
        )
        self._door_pos = torch.tensor(
            [layouts[c]["door_pos"] for c in codes],
            dtype=torch.float32,
            device=self.device,
        )
        self._n_layouts = len(codes)

        # Cached render grids (float for the dot, long for the wall mask).
        img = config.img_size
        lin = torch.linspace(0, img - 1, img, device=self.device)
        xx, yy = torch.meshgrid(lin, lin, indexing="xy")
        self._dot_grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        ar = torch.arange(0, img, device=self.device)
        gx, gy = torch.meshgrid(ar, ar, indexing="xy")
        self._wall_gx = gx.unsqueeze(0)  # (1, H, W)
        self._wall_gy = gy.unsqueeze(0)

    # ---- sync-free random samplers -------------------------------------------

    def _uniform(self, a, b):
        """Per-element uniform in [a, b]; a, b broadcastable tensors on device."""
        return a + (b - a) * torch.rand(
            torch.broadcast_shapes(a.shape, b.shape), device=self.device
        )

    def _truncated_normal(self, shape, mean, std, lo, hi):
        """Truncated normal via inverse CDF (exact, sync-free).

        Matches ``scipy.stats.truncnorm`` parameterisation used by the CPU
        generator: bounds ``lo, hi`` with underlying N(mean, std).
        """
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.device)
        lo = torch.as_tensor(lo, dtype=torch.float32, device=self.device)
        hi = torch.as_tensor(hi, dtype=torch.float32, device=self.device)

        def _phi(x):
            return 0.5 * (1.0 + torch.erf(x / _SQRT2))

        cdf_lo = _phi((lo - mean) / std)
        cdf_hi = _phi((hi - mean) / std)
        u = torch.rand(shape, device=self.device)
        p = cdf_lo + u * (cdf_hi - cdf_lo)
        p = p.clamp(1e-6, 1.0 - 1e-6)
        z = _SQRT2 * torch.erfinv(2.0 * p - 1.0)
        return mean + std * z

    def _von_mises(self, shape, kappa, rounds=24):
        """Vectorised Best-Fisher VonMises(loc=0, kappa) sampler, sync-free.

        Fixed number of rejection rounds so there is no data-dependent host
        loop. Acceptance prob is high (~0.65-0.9/round); after `rounds` the
        residual unfilled fraction is < 1e-9, those keep the fallback value 0.
        """
        kappa = float(kappa)
        tau = 1.0 + math.sqrt(1.0 + 4.0 * kappa * kappa)
        rho = (tau - math.sqrt(2.0 * tau)) / (2.0 * kappa)
        r = (1.0 + rho * rho) / (2.0 * rho)

        out = torch.zeros(shape, device=self.device)
        done = torch.zeros(shape, dtype=torch.bool, device=self.device)
        for _ in range(rounds):
            u1 = torch.rand(shape, device=self.device)
            u2 = torch.rand(shape, device=self.device)
            u3 = torch.rand(shape, device=self.device)
            z = torch.cos(math.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)
            accept = ((c * (2.0 - c) - u2) > 0) | ((torch.log(c / u2) + 1.0 - c) >= 0)
            theta = torch.sign(u3 - 0.5) * torch.acos(f.clamp(-1.0, 1.0))
            take = accept & (~done)
            out = torch.where(take, theta, out)
            done = done | accept
        return out

    # ---- layout / state / actions --------------------------------------------

    def _sample_walls(self, bs):
        idx = torch.randint(0, self._n_layouts, (bs,), device=self.device)
        return self._wall_pos[idx], self._door_pos[idx]

    def _generate_state(self, wall, door, bs):
        cfg = self.config
        pad = cfg.border_wall_loc - 1
        eff = (cfg.img_size - 1) - 2 * pad
        loc = torch.rand(bs, 2, device=self.device) * eff + pad

        hw = cfg.wall_width // 2
        left = wall - hw
        right = wall + hw
        dtop = door + cfg.door_space
        dbot = door - cfg.door_space

        btw = (loc[:, 0] >= left) & (loc[:, 0] <= right)
        not_btw_door = (loc[:, 1] < dbot) | (loc[:, 1] > dtop)
        inside = btw & not_btw_door

        minv = float(cfg.border_wall_loc - 1)
        maxv = float(cfg.img_size - cfg.border_wall_loc)
        change_left = torch.rand(bs, device=self.device) < 0.5
        new_left = self._uniform(torch.full((bs,), minv, device=self.device), left)
        new_right = self._uniform(right, torch.full((bs,), maxv, device=self.device))

        x = loc[:, 0]
        x = torch.where(inside & change_left, new_left, x)
        x = torch.where(inside & (~change_left), new_right, x)
        return torch.stack([x, loc[:, 1]], dim=1)

    def _generate_actions(self, bs, bias_angle=None):
        cfg = self.config
        n = cfg.n_steps
        if bias_angle is None:
            bias = torch.rand(bs, device=self.device) * 2.0 * math.pi
        else:
            bias = torch.atan2(bias_angle[:, 1], bias_angle[:, 0])

        kappa = 1.0 / cfg.action_angle_noise
        noise = self._von_mises((bs, n - 2), kappa)  # (bs, n-2)
        # cumulative angle walk; fmod(2pi) per-step in the CPU code is a no-op
        # because angles are only consumed through cos/sin.
        angles = torch.cat(
            [bias[:, None], bias[:, None] + noise.cumsum(dim=1)], dim=1
        )  # (bs, n-1)

        steps = self._truncated_normal(
            (bs, n - 1),
            cfg.action_step_mean,
            cfg.action_step_std,
            cfg.action_lower_bd,
            cfg.action_upper_bd,
        )
        vecs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        return vecs * steps.unsqueeze(-1)  # (bs, n-1, 2)

    def _generate_cross_wall(self, wall, door, bs):
        cfg = self.config
        n = cfg.n_steps
        hw = cfg.wall_width // 2
        left = wall - hw
        right = wall + hw

        x = self._uniform(left, right)
        y = self._truncated_normal(
            (bs,), door, 1.4, door - cfg.door_space, door + cfg.door_space
        )
        loc_at_door = torch.stack([x, y], dim=1)  # (bs, 2)

        step_idx = torch.randint(1, n, (bs,), device=self.device)  # [1, n-1]
        ang = math.pi + (torch.rand(bs, device=self.device) - 0.5) * math.pi / 2
        ang_vec = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        left_actions = self._generate_actions(bs, bias_angle=ang_vec)
        right_actions = self._generate_actions(bs, bias_angle=-ang_vec)

        # Build, per sample, traj = [flip(left[:step]) * -1, right[1 : n-step]].
        # Total length is always n-1 regardless of the per-sample split point.
        p = torch.arange(n - 1, device=self.device)  # (n-1,)
        is_A = p[None, :] < step_idx[:, None]  # (bs, n-1)
        idx_A = (step_idx[:, None] - 1 - p[None, :]).clamp(0, n - 2)
        idx_B = (p[None, :] - step_idx[:, None] + 1).clamp(0, n - 2)
        left_g = torch.gather(left_actions, 1, idx_A.unsqueeze(-1).expand(-1, -1, 2))
        right_g = torch.gather(right_actions, 1, idx_B.unsqueeze(-1).expand(-1, -1, 2))
        traj_base = torch.where(is_A.unsqueeze(-1), -left_g, right_g)  # (bs, n-1, 2)

        flip_mask = torch.rand(bs, device=self.device) < 0.5
        traj_flip = -torch.flip(traj_base, dims=[1])
        traj = torch.where(flip_mask[:, None, None], traj_flip, traj_base)

        # step_sum_before_door, computed on whichever orientation was kept.
        ssd_base = (traj_base * is_A.unsqueeze(-1)).sum(dim=1)  # (bs, 2)
        mask_flip = p[None, :] < (n - step_idx[:, None])
        ssd_flip = (traj_flip * mask_flip.unsqueeze(-1)).sum(dim=1)
        ssd = torch.where(flip_mask[:, None], ssd_flip, ssd_base)

        start = loc_at_door - ssd
        minv = cfg.border_wall_loc - 1 + 0.01
        maxv = cfg.img_size - cfg.border_wall_loc - 0.01
        start = start.clamp(minv, maxv)
        return start, traj

    # ---- collision physics (vectorised port of generate_transitions) ---------

    def _wall_intersection(self, cur, nxt, wall):
        hw = self.config.wall_width // 2
        wl = wall - hw
        wr = wall + hw
        cur_r = cur[:, 0] <= wr
        nxt_r = nxt[:, 0] <= wr
        cur_l = cur[:, 0] >= wl
        nxt_l = nxt[:, 0] >= wl
        inside = (cur_r & cur_l) != (nxt_r & nxt_l)
        across = (cur_r != nxt_r) & (cur_l != nxt_l)
        return inside | across

    @staticmethod
    def _seg_intersect(A, B):
        # A, B: (bs, 2, 2) — endpoints of two segments.
        A0, A1 = A[:, 0], A[:, 1]
        B0, B1 = B[:, 0], B[:, 1]
        dA = A1 - A0
        dB = B1 - B0

        def cross(v, w):
            return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

        iA = cross(dA, B0 - A0) * cross(dA, B1 - A0) < 0
        iB = cross(dB, A0 - B0) * cross(dB, A1 - B0) < 0
        return iA & iB

    def _wall_width_intersection(self, cur, nxt, wall, door):
        cfg = self.config
        disp = torch.stack([cur, nxt], dim=1)  # (bs, 2, 2)
        d = nxt - cur
        up = d[:, 1] > 0
        down = d[:, 1] < 0

        wl = wall - cfg.wall_width // 2
        wr = wall + cfg.wall_width // 2
        db = door - cfg.door_space
        dt = door + cfg.door_space

        tl = torch.stack([wl, dt], dim=1)
        tr = torch.stack([wr, dt], dim=1)
        bl = torch.stack([wl, db], dim=1)
        br = torch.stack([wr, db], dim=1)
        top_seg = torch.stack([tl, tr], dim=1)
        bot_seg = torch.stack([bl, br], dim=1)

        ti = self._seg_intersect(disp, top_seg)
        bi = self._seg_intersect(disp, bot_seg)
        return (ti & up) | (bi & down)

    def _pass_through_door(self, cur, nxt, wall, door):
        cfg = self.config
        hw = cfg.wall_width // 2
        lw = wall - hw
        rw = wall + hw
        d = nxt - cur
        a = d[:, 1] / d[:, 0]  # slope; nan/inf when dx==0 (masked out below)
        b = cur[:, 1] - a * cur[:, 0]
        db = door - cfg.door_space
        dt = door + cfg.door_space

        cross_l = torch.sign(lw - cur[:, 0]) * torch.sign(lw - nxt[:, 0]) < 0
        y_left = a * lw + b
        pass_left = (~cross_l) | ((db <= y_left) & (y_left <= dt))

        cross_r = torch.sign(rw - cur[:, 0]) * torch.sign(rw - nxt[:, 0]) < 0
        y_right = a * rw + b
        pass_right = (~cross_r) | ((db <= y_right) & (y_right <= dt))
        return pass_left & pass_right

    def _generate_transitions(self, loc, actions, wall, door):
        cfg = self.config
        bs = loc.shape[0]
        n_act = actions.shape[1]  # n_steps - 1

        lb = cfg.border_wall_loc - 1
        rb = cfg.img_size - cfg.border_wall_loc
        tb, bb = lb, rb

        cur = loc
        locs = [cur]
        for i in range(n_act):
            nxt = cur + actions[:, i]
            bx = (
                ((torch.sign(cur[:, 0] - lb) * torch.sign(nxt[:, 0] - lb)) <= 0)
                | ((torch.sign(cur[:, 0] - rb) * torch.sign(nxt[:, 0] - rb)) <= 0)
                | ((torch.sign(cur[:, 1] - tb) * torch.sign(nxt[:, 1] - tb)) <= 0)
                | ((torch.sign(cur[:, 1] - bb) * torch.sign(nxt[:, 1] - bb)) <= 0)
            )
            wint = self._wall_intersection(cur, nxt, wall)
            wwint = self._wall_width_intersection(cur, nxt, wall, door)
            passd = self._pass_through_door(cur, nxt, wall, door)
            block = bx | wwint | (wint & (~passd))
            nxt = torch.where(block.unsqueeze(-1), cur, nxt)
            locs.append(nxt)
            cur = nxt

        locations = torch.stack(locs, dim=1)  # (bs, n_steps, 2)

        # Render: dot channel (per frame) + static wall channel.
        states = self._render_location(locations.unsqueeze(-2))  # (bs, T, 1, H, W)
        walls = self._render_walls(wall, door).unsqueeze(1).unsqueeze(1)
        walls = walls.expand(-1, states.shape[1], -1, -1, -1)
        states = torch.cat([states, walls], dim=-3).float()  # (bs, T, 2, H, W)

        if cfg.normalize:
            states = self.normalizer.normalize_state(states)
            locations = self.normalizer.normalize_location(locations)

        # Drop the last frame so frames align with the n_steps-1 actions, then
        # sample, per sample, a contiguous window of length sample_length.
        states = states[:, :-1]  # (bs, n_steps-1, 2, H, W)
        locations = locations[:, :-1]  # (bs, n_steps-1, 2)

        sl = cfg.sample_length
        max_start = cfg.n_steps - sl  # exclusive upper bound, matches np.random.randint
        starts = torch.randint(0, max_start, (bs,), device=self.device)
        tidx = starts[:, None] + torch.arange(sl, device=self.device)[None, :]  # (bs,sl)
        b_ix = torch.arange(bs, device=self.device)[:, None]  # (bs, 1)

        states = states[b_ix, tidx].permute(0, 2, 1, 3, 4)  # (bs, 2, sl, H, W)
        actions_w = actions[b_ix, tidx].permute(0, 2, 1)  # (bs, 2, sl)
        locations = locations[b_ix, tidx].permute(0, 2, 1)  # (bs, 2, sl)

        return {
            "states": states,
            "actions": actions_w,
            "locations": locations,
            "wall_x": wall,
            "door_y": door,
        }

    # ---- rendering -----------------------------------------------------------

    def _render_location(self, locations):
        # locations: (..., 2) -> (..., H, W) uint8 gaussian dot
        lead = locations.shape[:-1]
        c = self._dot_grid.view(
            *([1] * len(lead)), *self._dot_grid.shape
        ).expand(*lead, *self._dot_grid.shape)
        loc = locations.unsqueeze(-2).unsqueeze(-2)  # (..., 1, 1, 2)
        d2 = (c - loc).pow(2).sum(dim=-1)  # squared distance (..., H, W)
        img = torch.exp(-d2 / (2.0 * self.config.dot_std * self.config.dot_std)) * 255.0
        return img.clamp(0, 255).to(torch.uint8)

    def _render_walls(self, wall, door):
        cfg = self.config
        bs = wall.shape[0]
        wr = wall.view(bs, 1, 1)
        dr = door.view(bs, 1, 1)
        off = cfg.wall_width // 2
        wall_mask = (wr - off <= self._wall_gx) & (self._wall_gx <= wr + off)
        res = (
            wall_mask
            & (
                (dr < self._wall_gy - cfg.door_space)
                | (dr > self._wall_gy + cfg.door_space)
            )
        ).float()
        bwl = cfg.border_wall_loc
        res[:, :, bwl - 1] = 1
        res[:, :, -bwl] = 1
        res[:, bwl - 1, :] = 1
        res[:, -bwl, :] = 1
        return (res * 255).clamp(0, 255).to(torch.uint8)

    # ---- public batch API ----------------------------------------------------

    def generate_batch(self, bs):
        """Generate ``bs`` independent samples on the current CUDA stream."""
        wall, door = self._sample_walls(bs)
        loc = self._generate_state(wall, door, bs)
        actions = self._generate_actions(bs, bias_angle=None)
        if self.config.cross_wall_rate > 0:
            cw_mask = torch.rand(bs, device=self.device) < self.config.cross_wall_rate
            cw_loc, cw_act = self._generate_cross_wall(wall, door, bs)
            loc = torch.where(cw_mask[:, None], cw_loc, loc)
            actions = torch.where(cw_mask[:, None, None], cw_act, actions)
        return self._generate_transitions(loc, actions, wall, door)
