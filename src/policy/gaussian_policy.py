"""Diagonal Gaussian MLP for GPS."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from src.policy import USE_ACT_NORM, featurize_obs, featurized_dim
from src.policy.ema import EMA
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

        # ---- Input pre-processor (see DeterministicPolicy for rationale) ----
        self._featurize_mode = getattr(cfg, "featurize", "running_norm")
        if self._featurize_mode == "hand_crafted":
            self.normalizer = None
            mlp_in = featurized_dim(obs_dim)
        else:
            self.normalizer = RunningNormalizer(obs_dim) if cfg.obs_norm else None
            mlp_in = obs_dim

        # ---- MLP stack with optional reg toggles ----
        use_dropout = getattr(cfg, "use_dropout", True)
        use_layernorm = getattr(cfg, "use_layernorm", True)
        dropout_p = getattr(cfg, "dropout_p", None)
        if dropout_p is None:
            dropout_p = 0.1   # legacy default for the Gaussian policy

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
        # (clamped to [log_sigma_min, log_sigma_max] in `_head`). When on,
        # `mu` is in normalized action space (-1, 1) and `_to_phys` maps
        # it to the physical range. NLL and PPO surrogates work unchanged
        # — they read `mu` and `log_sigma` independently.
        self._tanh_squash = bool(getattr(cfg, "tanh_squash", False))

        # Bounds are used to clip at execution time (act_np). When the
        # action-norm toggle is on, we additionally register a per-dim
        # affine (_act_scale, _act_bias) so the network learns/predicts in
        # normalized action space (~[-1, 1] per dim) and we denormalize at
        # the boundary. See `src/policy/__init__.py` for the full rationale.
        # `_has_act_norm` is the run-time check used by `_to_phys` /
        # `_to_norm` — set only when both the toggle is on AND we have
        # bounds to derive the affine from.
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

        # Optional EMA tracker over trainable parameters. Attached via
        # `attach_ema(decay)` from the trainer; when set, `train_weighted`
        # and `mse_step` call `ema.update(self)` after each Adam step so the
        # shadow weights track the training weights with low-pass smoothing.
        self.ema: EMA | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        self._device = next(self.parameters()).device
        return out

    # ------------------------------------------------------------------
    # Action-space (de)normalization helpers
    # ------------------------------------------------------------------
    # When the action-norm toggle is on, the network operates in the
    # normalized action space (~[-1, 1] per dim) and these convert at
    # the boundary. Both are no-ops when `_has_act_norm` is False, so
    # the toggle off path is byte-identical to the pre-norm code.

    def _to_phys(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map normalized network output to physical action space."""
        if not self._has_act_norm:
            return normalized
        return normalized * self._act_scale + self._act_bias

    def _to_norm(self, physical: torch.Tensor) -> torch.Tensor:
        """Map physical action to normalized action space."""
        if not self._has_act_norm:
            return physical
        return (physical - self._act_bias) / self._act_scale

    def _head(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply (featurize|normalizer) + MLP, return (mu, log_sigma) in
        NORMALIZED action space when USE_ACT_NORM is on, otherwise in
        physical space (legacy). When `tanh_squash` is on, `mu` is tanh-
        squashed to (-1, 1) before returning."""
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
        """obs: (B, obs_dim) → mu: (B, act_dim), log_sigma: (B, act_dim).

        Returns the network-internal head — same as `_head`. Callers that
        need the *physical* mean should use `action()` instead, which
        passes through `_to_phys`.
        """
        return self._head(obs)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π(actions | obs), shape (B,). Plain diagonal Gaussian.

        ``actions`` is taken in PHYSICAL space (matches the rest of the
        policy's external API and the action arrays MPPI hands in). When
        the action-norm toggle is on, we normalize the actions before
        evaluating the density against the network's normalized head; the
        Jacobian of the affine ``a = scale·n + bias`` is a constant per
        dim that cancels in any softmin / KL / gradient computation, so
        we omit it. Absolute log-prob values across runs with different
        toggle settings differ by `Σ log scale_i` — irrelevant for any
        downstream use here (MPPI prior, KL drift, NLL loss).
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
        KL(current || prev_policy).

        With uniform weights this reduces to plain NLL (what GPS and BC use).

        Args:
            prev_policy:   A frozen snapshot of this policy from the previous
                           GPS iteration (a deep copy, typically). When given
                           with `prev_kl_coef > 0`, the loss gains a term
                           `prev_kl_coef * E_o[ KL(π_θ(·|o) || π_prev(·|o)) ]`,
                           computed in closed form for two diagonal Gaussians.
                           This is the trust-region-style regularizer that
                           prevents the student from drifting too far from its
                           previous-iteration self in one S-step — stabilises
                           training when the C-step data shifts iter-to-iter.
            prev_kl_coef:  Penalty weight. 0 disables the KL term (default).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        # Labels arrive in physical action space (MPPI's executed actions).
        # Move them into the network's space so the loss compares like to
        # like. No-op when the toggle is off.
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
        # NaN-loss guard: a single batch containing NaN actions/obs
        # (e.g., from a divergent MPPI rollout in GPS C-step) would
        # otherwise produce NaN gradients on every parameter and
        # permanently corrupt the policy in one optimizer step. Skip
        # the step instead and surface NaN to the caller for logging.
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return float("nan")
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self)
        return loss.item()

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action tensor (policy mean) in PHYSICAL space.

        For MSE training / eval / mean-distance prior. Always returns a
        physical-space mean — this is the user-facing boundary, so the
        action-norm toggle is invisible to callers.
        """
        mu, _ = self.forward(obs)
        return self._to_phys(mu)

    def mse_step(self, obs: np.ndarray, actions: np.ndarray) -> float:
        """One gradient step of MSE on the mean. log_sigma is not supervised."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        # Same normalization as `train_weighted`: labels into the network's
        # space before the squared error, so per-dim contributions are equal.
        act_t = self._to_norm(act_t)
        if self.normalizer is not None:
            self.normalizer.update(obs_t)
        mu, _ = self.forward(obs_t)
        loss = ((mu - act_t) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        # See train_weighted: skip the step on a NaN/Inf loss so a single
        # bad batch can't corrupt every parameter via NaN gradients.
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return float("nan")
        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self)
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

        Closed-form KL for two diagonal Gaussians:

            KL = Σ_d [ log(σ_p / σ_q)
                     + (σ_q² + (μ_q − μ_p)²) / (2 σ_p²)
                     - 0.5 ]

        with subscript q = current policy, p = prev_policy. Per-dim sum,
        then mean over the batch.
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
        """Numpy-facing mean KL(self || prev_policy) over a batch of obs.

        Useful for logging iter-to-iter drift without touching the loss path.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        mu_q, log_sigma_q = self._head(obs_t)
        return float(self._kl_to_diag_gaussian(obs_t, mu_q, log_sigma_q, prev_policy).item())

    # ------------------------------------------------------------------
    # EMA hooks
    # ------------------------------------------------------------------

    def attach_ema(self, decay: float) -> None:
        """Start tracking an EMA of trainable parameters with given decay.

        Idempotent in the sense that calling it again resets the shadow to
        the current weights and switches to the new decay.
        """
        if decay <= 0.0:
            self.ema = None
            return
        self.ema = EMA(self, decay=decay)

    @contextmanager
    def ema_swapped_in(self):
        """Context manager: live params ← EMA shadow for the duration.

        No-op (yields immediately) when EMA is not attached.
        """
        if self.ema is None:
            yield
            return
        with self.ema.swapped_in(self):
            yield

    def ema_l2_drift(self) -> float:
        """||θ - θ_ema||₂. Returns 0.0 when EMA is not attached."""
        return self.ema.l2_drift(self) if self.ema is not None else 0.0

    def ema_sync(self) -> None:
        """Hard-sync: θ ← EMA shadow, in place. No-op if EMA not attached.

        Meant for end-of-S-step promotion in GPS / DAgger loops. Strongly
        recommended to follow this with `reset_optimizer()` so Adam's
        running moments don't carry stale gradients from the pre-sync θ.
        """
        if self.ema is not None:
            self.ema.sync_to(self)

    def reset_optimizer(self) -> None:
        """Recreate Adam with a fresh state (same lr). Clears m, v moments.

        Useful at GPS iteration boundaries when the effective loss surface
        shifts (new C-step data, new trust-region reference, a hard EMA
        sync that moved θ non-trivially). The alternative — letting stale
        momentum carry across iterations — fights the `prev_iter_kl_coef`
        trust region and mismatches Adam's state after an `ema_sync()`.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def log_prob_np(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        return self.log_prob(obs_t, act_t).cpu().numpy()

    @torch.no_grad()
    def act_np(self, obs: np.ndarray) -> np.ndarray:
        """Mean action from numpy obs, clipped to action bounds if known.

        Returns a physical-space action. Order: forward → denormalize →
        clip. The clip is still useful as a safety net even when the
        network was trained to output in [-1, 1] — the head is
        unconstrained (no tanh) so it can over/undershoot, and a clip
        keeps the env's `data.ctrl` in range.
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
