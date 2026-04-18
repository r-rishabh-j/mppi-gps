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

        if action_bounds is not None:
            low, high = action_bounds
            self.register_buffer("_act_low", torch.as_tensor(low, dtype=torch.float32))
            self.register_buffer("_act_high", torch.as_tensor(high, dtype=torch.float32))
        else:
            self._act_low = None
            self._act_high = None

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
        return self.net(x)

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
        a = self.forward(obs_t)
        if self._act_low is not None:
            a = torch.clamp(a, self._act_low, self._act_high)
        a_np = a.cpu().numpy()
        return a_np[0] if squeeze else a_np
