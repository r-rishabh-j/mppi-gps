"""Deterministic MLP policy: regresses actions directly (no mu/sigma head)."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from src.policy import USE_ACT_NORM
from src.policy.ema import EMA
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
            layers += [nn.Linear(in_dim, h), act_fn(),nn.LayerNorm(h), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

        # Action bounds + optional output-norm affine. See
        # `src/policy/__init__.py` and the GaussianPolicy mirror of this
        # block for the full rationale. `_has_act_norm` gates the
        # `_to_phys` / `_to_norm` helpers — set only when both the toggle
        # is on AND we have bounds to derive the affine from.
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

        # Optional EMA tracker — mirrors GaussianPolicy. KL-to-prev-iter is
        # not meaningful here (no policy distribution), so only EMA is wired.
        self.ema: EMA | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        self._device = next(self.parameters()).device
        return out

    # ------------------------------------------------------------------
    # Action-space (de)normalization helpers (mirror GaussianPolicy)
    # ------------------------------------------------------------------

    def _to_phys(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self._has_act_norm:
            return normalized
        return normalized * self._act_scale + self._act_bias

    def _to_norm(self, physical: torch.Tensor) -> torch.Tensor:
        if not self._has_act_norm:
            return physical
        return (physical - self._act_bias) / self._act_scale

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Network-internal output: NORMALIZED action space when the toggle
        is on, physical space otherwise. Use `action()` for a physical-
        space tensor at the user-facing boundary."""
        x = self.normalizer(obs) if self.normalizer is not None else obs
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
        """One MSE update step. When ``grad_clip_norm > 0``, the parameter
        gradient is L2-norm-clipped to that bound between ``backward()`` and
        ``optimizer.step()`` — a soft trust region on per-update parameter
        movement that is independent of the loss magnitude or the residual
        distribution. Loss-agnostic, dimension-aware, no biased estimator
        (unlike target-clipping the labels)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        # Move physical labels into the network's space so per-dim losses
        # are equalized — see `src/policy/__init__.py`. No-op when the
        # action-norm toggle is off.
        act_t = self._to_norm(act_t)
        if self.normalizer is not None:
            self.normalizer.update(obs_t)
        pred = self.forward(obs_t)
        loss = ((pred - act_t) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        # NaN-loss guard: see GaussianPolicy.train_weighted. A single bad
        # batch (NaN action target from a divergent MPPI rollout) would
        # otherwise overwrite every parameter with NaN in one step.
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return float("nan")
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self)
        return float(loss.item())

    # ------------------------------------------------------------------
    # EMA hooks (mirror GaussianPolicy)
    # ------------------------------------------------------------------

    def attach_ema(self, decay: float) -> None:
        if decay <= 0.0:
            self.ema = None
            return
        self.ema = EMA(self, decay=decay)

    @contextmanager
    def ema_swapped_in(self):
        if self.ema is None:
            yield
            return
        with self.ema.swapped_in(self):
            yield

    def ema_l2_drift(self) -> float:
        return self.ema.l2_drift(self) if self.ema is not None else 0.0

    def ema_sync(self) -> None:
        """Hard-sync θ ← EMA shadow. See GaussianPolicy.ema_sync()."""
        if self.ema is not None:
            self.ema.sync_to(self)

    def reset_optimizer(self) -> None:
        """Recreate Adam with a fresh state. See GaussianPolicy.reset_optimizer()."""
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
