"""Diagonal Gaussian MLP for GPS."""

import numpy as np
import torch
import torch.nn as nn

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

        self.normalizer = RunningNormalizer(obs_dim) if cfg.obs_norm else None

        layers = []
        in_dim = obs_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 2 * act_dim))
        self.net = nn.Sequential(*layers)

        # Init log_sigma head bias to 0 → sigma ≈ 1 at start.
        nn.init.zeros_(self.net[-1].bias[act_dim:])

        # Tanh-squash: store action scale/bias so mu lands in [low, high].
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

    def _head(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply normalizer + MLP, return raw (mu_pre, log_sigma) — unsquashed."""
        x = self.normalizer(obs) if self.normalizer is not None else obs
        out = self.net(x)
        mu_pre, log_sigma = out[..., :self.act_dim], out[..., self.act_dim:]
        log_sigma = log_sigma.clamp(self.cfg.log_sigma_min, self.cfg.log_sigma_max)
        return mu_pre, log_sigma

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, obs_dim) → mu: (B, act_dim), log_sigma: (B, act_dim).

        When squash_tanh is on, mu is the squashed mean in action space. The
        log_sigma returned here is still in pre-tanh (base-Gaussian) units —
        use log_prob / sample to get correctly-corrected values.
        """
        mu_pre, log_sigma = self._head(obs)
        if self._squash:
            mu = torch.tanh(mu_pre) * self._act_scale + self._act_bias
        else:
            mu = mu_pre
        return mu, log_sigma

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """log π(actions | obs), shape (B,). Applies tanh change-of-vars if enabled."""
        mu_pre, log_sigma = self._head(obs)
        sigma = log_sigma.exp()

        if self._squash:
            # Invert the squash: u_pre such that action = tanh(u_pre)*scale + bias.
            # Numerically stable atanh via clamp.
            norm = (actions - self._act_bias) / self._act_scale
            norm = norm.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            u_pre = torch.atanh(norm)
            base = -0.5 * (((u_pre - mu_pre) / sigma) ** 2 + 2 * log_sigma + np.log(2 * np.pi))
            # Change-of-variables: subtract log|d tanh / d u| = log(1 - tanh²),
            # plus the linear rescaling log(scale).
            jac = torch.log(1.0 - torch.tanh(u_pre) ** 2 + 1e-6) + torch.log(self._act_scale + 1e-12)
            return (base - jac).sum(dim=-1)

        lp = -0.5 * (((actions - mu_pre) / sigma) ** 2 + 2 * log_sigma + np.log(2 * np.pi))
        return lp.sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample of an action."""
        mu_pre, log_sigma = self._head(obs)
        u_pre = mu_pre + log_sigma.exp() * torch.randn_like(mu_pre)
        if self._squash:
            return torch.tanh(u_pre) * self._act_scale + self._act_bias
        return u_pre

    def train_weighted(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """One gradient step of weighted NLL.

        With uniform weights this reduces to plain NLL (what GPS and BC use).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        w_t = torch.as_tensor(weights, dtype=torch.float32, device=self._device)

        if self.normalizer is not None:
            self.normalizer.update(obs_t)

        lp = self.log_prob(obs_t, act_t)
        loss = -(w_t * lp).sum() / w_t.sum().clamp(min=1e-8)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action tensor (policy mean). For MSE training / eval."""
        mu, _ = self.forward(obs)
        return mu

    def mse_step(self, obs: np.ndarray, actions: np.ndarray) -> float:
        """One gradient step of MSE loss on the mean (deterministic regression).

        log_sigma is not supervised; only mu gets gradient. Used by DAgger.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if self.normalizer is not None:
            self.normalizer.update(obs_t)
        mu, _ = self.forward(obs_t)
        loss = ((mu - act_t) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def log_prob_np(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        return self.log_prob(obs_t, act_t).cpu().numpy()

    @torch.no_grad()
    def act_np(self, obs: np.ndarray) -> np.ndarray:
        """Mean action from numpy obs. Used by eval / MPPI warm-start."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        squeeze = obs_t.ndim == 1
        if squeeze:
            obs_t = obs_t.unsqueeze(0)
        mu, _ = self.forward(obs_t)
        mu_np = mu.cpu().numpy()
        return mu_np[0] if squeeze else mu_np
