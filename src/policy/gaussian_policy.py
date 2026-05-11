"""Diagonal Gaussian MLP for GPS."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.policy import USE_ACT_NORM, featurize_obs, featurized_dim
from src.utils.config import PolicyConfig


class RunningNormalizer(nn.Module):
    """Running mean/var over inputs. Buffers save via state_dict.

    update() is explicit: only the supervised training path calls it.
    forward() always normalizes with the current frozen stats.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(1))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # Welford parallel merge of batch stats into running stats.
        x = x.reshape(-1, x.shape[-1])
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total
        new_var = M2 / total

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian MLP: obs → (mu, log_sigma)."""

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

        # Input pre-processor: hand-crafted featurization bypasses
        # RunningNormalizer (already produces unit-scaled features).
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
            dropout_p = 0.1

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
        layers.append(nn.Linear(in_dim, 2 * act_dim))
        self.net = nn.Sequential(*layers)

        # Init log_sigma head bias to 0 → sigma ≈ 1 at start.
        nn.init.zeros_(self.net[-1].bias[act_dim:])

        # Optional tanh on the mu head only. log_sigma stays unbounded
        # (clamped in `_head`). With tanh on, mu lives in normalized
        # action space (-1, 1) and `_to_phys` denormalizes.
        self._tanh_squash = bool(getattr(cfg, "tanh_squash", False))

        # Action bounds clamp at execution time. When USE_ACT_NORM is on,
        # also register a per-dim affine so the head trains/predicts in
        # normalized action space and we denormalize at the boundary.
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

    # Action-space (de)normalization. No-ops when `_has_act_norm` is False.

    def _to_phys(self, normalized: torch.Tensor) -> torch.Tensor:
        if not self._has_act_norm:
            return normalized
        return normalized * self._act_scale + self._act_bias

    def _to_norm(self, physical: torch.Tensor) -> torch.Tensor:
        if not self._has_act_norm:
            return physical
        return (physical - self._act_bias) / self._act_scale

    def _head(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(featurize|normalizer) + MLP → (mu, log_sigma). mu is in
        NORMALIZED action space when USE_ACT_NORM is on (physical
        otherwise); tanh-squashed when `tanh_squash` is True."""
        if self._featurize_mode == "hand_crafted":
            x = featurize_obs(obs)
        elif self.normalizer is not None:
            x = self.normalizer(obs)
        else:
            x = obs
        out = self.net(x)
        mu, log_sigma = out[..., :self.act_dim], out[..., self.act_dim:]
        if self._tanh_squash:
            mu = torch.tanh(mu)
        log_sigma = log_sigma.clamp(self.cfg.log_sigma_min, self.cfg.log_sigma_max)
        return mu, log_sigma

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the network-internal head — same as `_head`. Use
        `action()` for a physical-space mean."""
        return self._head(obs)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π(actions | obs), shape (B,). Diagonal Gaussian.

        `actions` is in PHYSICAL space; we normalize before evaluating
        against the head. The Jacobian of the affine `a = scale·n + bias`
        is constant per dim and cancels in any softmin / KL / gradient,
        so we omit it.
        """
        actions = self._to_norm(actions)
        mu, log_sigma = self._head(obs)
        sigma = log_sigma.exp()
        lp = -0.5 * (((actions - mu) / sigma) ** 2 + 2 * log_sigma + np.log(2 * np.pi))
        return lp.sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample of an action, in PHYSICAL space."""
        mu, log_sigma = self._head(obs)
        n = mu + log_sigma.exp() * torch.randn_like(mu)
        return self._to_phys(n)

    def train_weighted(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
        prev_policy: "GaussianPolicy | None" = None,
        prev_kl_coef: float = 0.0,
    ) -> float:
        """One gradient step of weighted NLL, optionally regularized by
        KL(current || prev_policy). With uniform weights this reduces to
        plain NLL (what GPS and BC use).

        When `prev_policy` is given with `prev_kl_coef > 0`, the loss
        gains `prev_kl_coef * E_o[KL(π_θ || π_prev)]` — a trust-region
        regularizer against the previous-iteration policy.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        act_t = self._to_norm(act_t)
        w_t = torch.as_tensor(weights, dtype=torch.float32, device=self._device)

        if self.normalizer is not None:
            self.normalizer.update(obs_t)

        # Single forward pass — reuse mu/log_sigma for both log_prob and KL.
        mu, log_sigma = self._head(obs_t)
        sigma = log_sigma.exp()
        lp = -0.5 * (
            ((act_t - mu) / sigma) ** 2
            + 2.0 * log_sigma
            + np.log(2.0 * np.pi)
        )
        lp = lp.sum(dim=-1)
        loss = -(w_t * lp).sum() / w_t.sum().clamp(min=1e-8)

        if prev_policy is not None and prev_kl_coef > 0.0:
            loss = loss + prev_kl_coef * self._kl_to_diag_gaussian(
                obs_t, mu, log_sigma, prev_policy,
            )

        self.optimizer.zero_grad()
        loss.backward()
        # NaN guard: a NaN batch would corrupt every parameter via NaN
        # gradients in one step. Skip and surface NaN to the caller.
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return float("nan")
        self.optimizer.step()
        return loss.item()

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (policy mean) in PHYSICAL space."""
        mu, _ = self.forward(obs)
        return self._to_phys(mu)

    def mse_step(self, obs: np.ndarray, actions: np.ndarray) -> float:
        """One gradient step of MSE on the mean. log_sigma is not supervised."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        act_t = self._to_norm(act_t)
        if self.normalizer is not None:
            self.normalizer.update(obs_t)
        mu, _ = self.forward(obs_t)
        loss = ((mu - act_t) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return float("nan")
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # KL helpers
    # ------------------------------------------------------------------

    def _kl_to_diag_gaussian(
        self,
        obs_t: torch.Tensor,
        mu_q: torch.Tensor,
        log_sigma_q: torch.Tensor,
        prev_policy: "GaussianPolicy",
    ) -> torch.Tensor:
        """Mean KL(π_θ(·|o) || π_prev(·|o)) over the batch.

        Closed-form for two diagonal Gaussians:
            KL = Σ_d [log(σ_p/σ_q) + (σ_q² + (μ_q − μ_p)²)/(2 σ_p²) − 0.5]
        """
        with torch.no_grad():
            mu_p, log_sigma_p = prev_policy._head(obs_t)
        var_q = torch.exp(2.0 * log_sigma_q)
        var_p = torch.exp(2.0 * log_sigma_p)
        kl_per_dim = (
            log_sigma_p - log_sigma_q
            + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p)
            - 0.5
        )
        return kl_per_dim.sum(dim=-1).mean()

    @torch.no_grad()
    def kl_to_np(
        self,
        obs: np.ndarray,
        prev_policy: "GaussianPolicy",
    ) -> float:
        """Numpy-facing mean KL(self || prev_policy) over a batch of obs."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        mu_q, log_sigma_q = self._head(obs_t)
        return float(self._kl_to_diag_gaussian(obs_t, mu_q, log_sigma_q, prev_policy).item())

    def reset_optimizer(self) -> None:
        """Recreate Adam with a fresh state (same lr). Clears m, v moments.

        Useful at GPS iteration boundaries — each iter is a new supervised
        task (new C-step data + prior shift), and stale momentum can fight
        the `prev_iter_kl_coef` trust region.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def log_prob_np(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        return self.log_prob(obs_t, act_t).cpu().numpy()

    @torch.no_grad()
    def act_np(self, obs: np.ndarray) -> np.ndarray:
        """Mean action from numpy obs, clipped to action bounds.

        Order: forward → denormalize → clip. The clip is still a useful
        safety net when the head is unconstrained (no tanh).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        squeeze = obs_t.ndim == 1
        if squeeze:
            obs_t = obs_t.unsqueeze(0)
        mu, _ = self.forward(obs_t)
        mu = self._to_phys(mu)
        if self._act_low is not None:
            mu = torch.clamp(mu, self._act_low, self._act_high)
        mu_np = mu.cpu().numpy()
        return mu_np[0] if squeeze else mu_np
