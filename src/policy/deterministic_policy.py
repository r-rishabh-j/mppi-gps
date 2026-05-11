"""Deterministic MLP policy: regresses actions directly (no mu/sigma head)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.policy import USE_ACT_NORM, featurize_obs, featurized_dim
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

        # Hand-crafted featurization bypasses RunningNormalizer.
        self._featurize_mode = getattr(cfg, "featurize", "running_norm")
        if self._featurize_mode == "hand_crafted":
            self.normalizer = None
            mlp_in = featurized_dim(obs_dim)
        else:
            self.normalizer = RunningNormalizer(obs_dim) if cfg.obs_norm else None
            mlp_in = obs_dim

        use_dropout = getattr(cfg, "use_dropout", True)
        use_layernorm = getattr(cfg, "use_layernorm", True)
        dropout_p = getattr(cfg, "dropout_p", None)
        if dropout_p is None:
            dropout_p = 0.2

        layers = []
        in_dim = mlp_in
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            if use_dropout:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        # Tanh squash: network output bounded to (-1, 1) in normalized
        # action space; `_to_phys` denormalizes.
        self._tanh_squash = bool(getattr(cfg, "tanh_squash", False))
        if self._tanh_squash:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        # Bounds + optional output-norm affine (mirrors GaussianPolicy).
        self._has_act_norm = False
        if action_bounds is not None:
            low, high = action_bounds
            self.register_buffer("_act_low", torch.as_tensor(low, dtype=torch.float32))
            self.register_buffer("_act_high", torch.as_tensor(high, dtype=torch.float32))
            if USE_ACT_NORM:
                scale = (high - low) / 2.0
                bias = (high + low) / 2.0
                self.register_buffer(
                    "_act_scale", torch.as_tensor(scale, dtype=torch.float32)
                )
                self.register_buffer(
                    "_act_bias", torch.as_tensor(bias, dtype=torch.float32)
                )
                self._has_act_norm = True
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

    # Action-space (de)normalization (mirror GaussianPolicy).

    def _to_phys(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self._has_act_norm:
            return normalized
        return normalized * self._act_scale + self._act_bias

    def _to_norm(self, physical: torch.Tensor) -> torch.Tensor:
        if not self._has_act_norm:
            return physical
        return (physical - self._act_bias) / self._act_scale

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Network output in NORMALIZED action space when the affine
        toggle is on, physical otherwise. Use `action()` at the user
        boundary."""
        if self._featurize_mode == "hand_crafted":
            x = featurize_obs(obs)
        elif self.normalizer is not None:
            x = self.normalizer(obs)
        else:
            x = obs
        return self.net(x)

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action tensor in PHYSICAL space."""
        return self._to_phys(self.forward(obs))

    def mse_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        grad_clip_norm: float = 0.0,
    ) -> float:
        """One MSE update. `grad_clip_norm > 0` activates an L2 grad-norm
        clip between backward() and optimizer.step() — a loss-agnostic
        bound on per-update parameter movement."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        act_t = self._to_norm(act_t)
        if self.normalizer is not None:
            self.normalizer.update(obs_t)
        pred = self.forward(obs_t)
        loss = ((pred - act_t) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return float("nan")
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    def reset_optimizer(self) -> None:
        """Recreate Adam with fresh state. See GaussianPolicy.reset_optimizer."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def act_np(self, obs: np.ndarray) -> np.ndarray:
        """Physical-space action. Order: forward → denormalize → clip."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        squeeze = obs_t.ndim == 1
        if squeeze:
            obs_t = obs_t.unsqueeze(0)
        a = self.forward(obs_t)
        a = self._to_phys(a)
        if self._act_low is not None:
            a = torch.clamp(a, self._act_low, self._act_high)
        a_np = a.cpu().numpy()
        return a_np[0] if squeeze else a_np
