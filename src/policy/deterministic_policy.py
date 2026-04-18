"""Deterministic MLP policy: regresses actions directly (no mu/sigma head)."""

import numpy as np
import torch
import torch.nn as nn

from src.policy.gaussian_policy import RunningNormalizer
from src.utils.config import PolicyConfig


class DeterministicPolicy(nn.Module):
    """obs → action. MSE regression target, no log_prob."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: PolicyConfig = PolicyConfig(),
        device: torch.device | str | None = None,
        action_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg

        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        act_fn = activations[cfg.activation]

        self.normalizer = RunningNormalizer(obs_dim) if cfg.obs_norm else None

        layers = []
        in_dim = obs_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

        self._squash = bool(cfg.squash_tanh)
        if self._squash:
            if action_bounds is None:
                raise ValueError("squash_tanh=True requires action_bounds")
            low, high = action_bounds
            low_t = torch.as_tensor(low, dtype=torch.float32)
            high_t = torch.as_tensor(high, dtype=torch.float32)
            self.register_buffer("_act_scale", (high_t - low_t) / 2.0)
            self.register_buffer("_act_bias", (high_t + low_t) / 2.0)

        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self.to(self._device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        self._device = next(self.parameters()).device
        return out

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(obs) if self.normalizer is not None else obs
        a = self.net(x)
        if self._squash:
            a = torch.tanh(a) * self._act_scale + self._act_bias
        return a

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    def mse_step(self, obs: np.ndarray, actions: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if self.normalizer is not None:
            self.normalizer.update(obs_t)
        pred = self.forward(obs_t)
        loss = ((pred - act_t) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def act_np(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        squeeze = obs_t.ndim == 1
        if squeeze:
            obs_t = obs_t.unsqueeze(0)
        a = self.forward(obs_t).cpu().numpy()
        return a[0] if squeeze else a
