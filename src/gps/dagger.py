"""DAgger distillation of MPPI into a GaussianPolicy / DeterministicPolicy.

Per iteration: roll out the current policy (β-mixed with MPPI), query
MPPI as the expert at every visited state, append to an aggregating
buffer, and finetune.

MPPI on CPU; policy training on the policy's device.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import DAggerConfig


class DAggerTrainer:
    def __init__(
        self,
        env: BaseEnv,
        mppi: MPPI,
        policy: GaussianPolicy,
        cfg: DAggerConfig,
        rng: np.random.Generator | None = None,
    ):
        self.env = env
        self.mppi = mppi
        self.policy = policy
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)

        self.obs_dim = policy.obs_dim
        self.act_dim = policy.act_dim

        self._buf_obs: list[np.ndarray] = []
        self._buf_act: list[np.ndarray] = []
        self._buf_round: list[np.ndarray] = []

    def buffer_size(self) -> int:
        return sum(len(o) for o in self._buf_obs)

    def append(self, obs: np.ndarray, act: np.ndarray, round_idx: int) -> None:
        self._buf_obs.append(obs.astype(np.float32))
        self._buf_act.append(act.astype(np.float32))
        self._buf_round.append(np.full(len(obs), round_idx, dtype=np.int32))
        self._evict_if_full()

    def _evict_if_full(self) -> None:
        cap = self.cfg.buffer_cap
        total = self.buffer_size()
        while total > cap and len(self._buf_obs) > 1:
            drop = len(self._buf_obs[0])
            self._buf_obs.pop(0)
            self._buf_act.pop(0)
            self._buf_round.pop(0)
            total -= drop

    def flat_buffer(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = np.concatenate(self._buf_obs, axis=0)
        act = np.concatenate(self._buf_act, axis=0)
        rnd = np.concatenate(self._buf_round, axis=0)
        return obs, act, rnd

    def seed_from_h5(self, path) -> None:
        """Warm-start the buffer from an existing BC dataset."""
        import h5py

        with h5py.File(path, "r") as f:
            s = f["states"][:].astype(np.float32)  # (M, T, obs_dim)
            a = f["actions"][:].astype(np.float32)  # (M, T, act_dim)
        self.append(s.reshape(-1, s.shape[-1]), a.reshape(-1, a.shape[-1]), round_idx=-1)

    def beta(self, k: int) -> float:
        K = self.cfg.dagger_iters
        if self.cfg.beta_schedule == "constant_zero":
            return 0.0
        # Linear: 1.0 at k=0, 0.0 at k≥K/2.
        half = max(1, K // 2)
        return max(0.0, 1.0 - k / half)

    def _collect(
        self,
        n_rollouts: int,
        beta: float,
        seed_base: int,
        episode_len: int | None = None,
        progress_label: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out under β, return flat (obs, expert_actions)."""
        ep_len = episode_len if episode_len is not None else self.cfg.episode_len
        obs_rows: list[np.ndarray] = []
        act_rows: list[np.ndarray] = []

        auto_reset = getattr(self.cfg, "auto_reset", False)
        total_steps = n_rollouts * ep_len
        with tqdm(
            total=total_steps,
            desc=progress_label,
            leave=False,
            dynamic_ncols=True,
            disable=progress_label is None,
        ) as pbar:
            for ep in range(n_rollouts):
                np.random.seed(seed_base + ep)
                self.env.reset()
                self.mppi.reset()

                for _ in range(ep_len):
                    obs = self.env._get_obs()
                    state = self.env.get_state()
                    expert_action, _ = self.mppi.plan_step(state, dry_run=False)

                    if self.rng.random() < beta:
                        exec_action = expert_action
                    else:
                        exec_action = self.policy.act_np(obs)

                    obs_rows.append(obs.astype(np.float32))
                    act_rows.append(expert_action.astype(np.float32))

                    _, _, done, _ = self.env.step(exec_action)
                    pbar.update(1)
                    if done:
                        # auto_reset: keep collecting up to ep_len (for
                        # terminating envs). Otherwise break.
                        if auto_reset:
                            self.env.reset()
                            self.mppi.reset()
                        else:
                            break

        obs_arr = np.stack(obs_rows, axis=0)
        act_arr = np.stack(act_rows, axis=0)
        if act_arr.ndim == 1:
            act_arr = act_arr[:, None]
        return obs_arr, act_arr

    def collect_round(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Roll out ``rollouts_per_iter`` trajs with β-mix; relabel every state."""
        return self._collect(
            n_rollouts=self.cfg.rollouts_per_iter,
            beta=self.beta(k),
            seed_base=self.cfg.seed + 1000 * k,
            progress_label=f"round {k} rollout",
        )

    def warmup(
        self,
        n_rollouts: int,
        epochs: int,
        cache_path: "Path | str | None" = None,
    ) -> list[float]:
        """Pre-train on pure-MPPI rollouts; appends with round_idx=-1.

        ``cache_path`` (optional) loads if present (auto-flattens BC-style
        ``(M, T, dim)`` h5), else collects + saves with the
        ``collect_bc_demos`` schema. Returns per-epoch train losses.
        """
        if n_rollouts <= 0 or epochs <= 0:
            return []

        cache_path = Path(cache_path) if cache_path is not None else None
        obs_r: np.ndarray
        act_r: np.ndarray

        if cache_path is not None and cache_path.exists():
            import h5py

            with h5py.File(cache_path, "r") as f:
                obs_r = f["states"][:].astype(np.float32)
                act_r = f["actions"][:].astype(np.float32)
                cached_n = int(f.attrs.get("n_rollouts", -1))
                cached_len = int(f.attrs.get("episode_len", -1))
            # Flatten BC-style (M, T, dim) → (N, dim).
            if obs_r.ndim == 3:
                obs_r = obs_r.reshape(-1, obs_r.shape[-1])
                act_r = act_r.reshape(-1, act_r.shape[-1])
            print(f"  loaded {len(obs_r):,} warmup rows from cache {cache_path}")
            if cached_n > 0 and (
                cached_n != n_rollouts or cached_len != self.cfg.episode_len
            ):
                print(
                    f"  warning: cache was collected with n_rollouts={cached_n}, "
                    f"episode_len={cached_len}; you requested "
                    f"{n_rollouts}/{self.cfg.episode_len}. Using the cache as-is — "
                    f"delete the file to recollect."
                )
        else:
            obs_r, act_r = self._collect(
                n_rollouts=n_rollouts, beta=1.0,
                seed_base=self.cfg.seed + 999_000,  # distinct from round seeds
                progress_label="warmup rollout",
            )
            if cache_path is not None:
                import h5py

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(cache_path, "w") as f:
                    f.create_dataset("states", data=obs_r)
                    f.create_dataset("actions", data=act_r)
                    f.attrs["n_rollouts"] = n_rollouts
                    f.attrs["episode_len"] = self.cfg.episode_len
                    f.attrs["auto_reset"] = bool(getattr(self.cfg, "auto_reset", False))
                    f.attrs["seed"] = self.cfg.seed
                    f.attrs["obs_dim"] = self.obs_dim
                    f.attrs["act_dim"] = self.act_dim
                print(f"  saved {len(obs_r):,} warmup rows to {cache_path}")

        self.append(obs_r, act_r, round_idx=-1)

        N = len(obs_r)
        idx = np.arange(N)
        losses: list[float] = []
        grad_clip_norm = float(getattr(self.cfg, "grad_clip_norm", 0.0))
        # PPO clip skipped here — random-init ratio is meaningless.
        epoch_bar = tqdm(
            range(epochs),
            desc="warmup fit",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in epoch_bar:
            self.rng.shuffle(idx)
            running, nb = 0.0, 0
            for start in range(0, N, self.cfg.batch_size):
                b = idx[start:start + self.cfg.batch_size]
                running += self._train_step(
                    obs_r[b], act_r[b], grad_clip_norm=grad_clip_norm,
                )
                nb += 1
            loss = running / max(nb, 1)
            losses.append(loss)
            epoch_bar.set_postfix(train_loss=f"{loss:.4f}")
        return losses

    def _train_step(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        grad_clip_norm: float = 0.0,
        old_policy: "GaussianPolicy | None" = None,
        clip_ratio: float = 0.2,
    ) -> float:
        """One distillation step.

        - Deterministic: MSE on the mean (``grad_clip_norm`` activates L2 clip).
        - Gaussian + ``old_policy`` + ``clip_ratio > 0``: PPO-clip surrogate.
        - Gaussian default: full-diagonal NLL via ``train_weighted`` with
          uniform weights (trains both mu and log_sigma).
        """
        if isinstance(self.policy, DeterministicPolicy):
            if grad_clip_norm > 0.0:
                return self.policy.mse_step(obs, act, grad_clip_norm=grad_clip_norm)
            return self.policy.mse_step(obs, act)

        if (
            isinstance(self.policy, GaussianPolicy)
            and old_policy is not None
            and clip_ratio > 0.0
        ):
            return self._train_step_gaussian_ppo(obs, act, old_policy, clip_ratio)

        weights = np.ones(len(obs), dtype=np.float32)
        return self.policy.train_weighted(obs, act, weights)

    def _train_step_gaussian_ppo(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        old_policy: GaussianPolicy,
        clip_ratio: float,
    ) -> float:
        """PPO ratio-clip surrogate, advantage=1.

        ``L = -E[min(r, clip(r, 1-ε, 1+ε))]``,
        ``r = π_θ(a|o) / π_old(a|o)``.
        """
        device = self.policy.device
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(act, dtype=torch.float32, device=device)

        if self.policy.normalizer is not None:
            self.policy.normalizer.update(o_t)

        curr_log_prob = self.policy.log_prob(o_t, a_t)
        with torch.no_grad():
            old_log_prob = old_policy.log_prob(o_t, a_t)

        # Clamp log-ratio so exp() can't overflow into NaN gradients.
        log_ratio = torch.clamp(curr_log_prob - old_log_prob, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        loss = -torch.min(surr1, surr2).mean()

        self.policy.optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss):
            self.policy.optimizer.zero_grad()
            return float("nan")
        self.policy.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def _eval_mse(self, obs: np.ndarray, act: np.ndarray, batch: int = 16384) -> float:
        """Validation MSE in the policy's training-loss space (normalized
        when output norm is on; physical otherwise)."""
        device = self.policy.device
        total, n = 0.0, 0
        for s in range(0, len(obs), batch):
            o = torch.as_tensor(obs[s:s + batch], dtype=torch.float32, device=device)
            a = torch.as_tensor(act[s:s + batch], dtype=torch.float32, device=device)
            pred = self.policy.action(o)            # physical
            diff = pred - a
            if self.policy._has_act_norm:
                diff = diff / self.policy._act_scale  # → normalized space
            total += float((diff ** 2).sum().item())
            n += a.numel()
        return total / max(n, 1)

    def finetune(
        self,
        latest_obs: np.ndarray,
        latest_act: np.ndarray,
        round_idx: int | None = None,
    ) -> tuple[float, float]:
        """Finetune on the full buffer; val uses a slice of the latest round."""
        n_latest = len(latest_obs)
        perm = self.rng.permutation(n_latest)
        n_val = max(1, int(n_latest * self.cfg.val_frac))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        va_o, va_a = latest_obs[val_idx], latest_act[val_idx]

        # Full training set = buffer minus the val slice from latest round.
        all_o, all_a, _ = self.flat_buffer()
        n_before_latest = self.buffer_size() - n_latest
        latest_train_o = latest_obs[train_idx]
        latest_train_a = latest_act[train_idx]
        tr_o = np.concatenate([all_o[:n_before_latest], latest_train_o], axis=0)
        tr_a = np.concatenate([all_a[:n_before_latest], latest_train_a], axis=0)

        N = len(tr_o)
        idx = np.arange(N)
        last_train = last_val = float("nan")
        grad_clip_norm = float(getattr(self.cfg, "grad_clip_norm", 0.0))
        clip_ratio = float(getattr(self.cfg, "clip_ratio", 0.0))

        # Pre-finetune snapshot for the PPO clip (Gaussian + clip_ratio>0).
        old_policy = None
        if isinstance(self.policy, GaussianPolicy) and clip_ratio > 0.0:
            old_policy = copy.deepcopy(self.policy)
            old_policy.eval()
            for p in old_policy.parameters():
                p.requires_grad_(False)

        epoch_bar = tqdm(
            range(self.cfg.distill_epochs),
            desc=f"round {round_idx} fit" if round_idx is not None else "dagger fit",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in epoch_bar:
            self.rng.shuffle(idx)
            running, nb = 0.0, 0
            for start in range(0, N, self.cfg.batch_size):
                b = idx[start:start + self.cfg.batch_size]
                running += self._train_step(
                    tr_o[b], tr_a[b],
                    grad_clip_norm=grad_clip_norm,
                    old_policy=old_policy,
                    clip_ratio=clip_ratio,
                )
                nb += 1
            last_train = running / max(nb, 1)
            last_val = self._eval_mse(va_o, va_a)
            epoch_bar.set_postfix(train_loss=f"{last_train:.4f}", val_mse=f"{last_val:.4f}")
        return last_train, last_val

    def step(self, k: int) -> dict:
        obs_r, act_r = self.collect_round(k)
        self.append(obs_r, act_r, round_idx=k)
        train_loss, val_mse = self.finetune(obs_r, act_r, round_idx=k)
        if getattr(self.cfg, "reset_optim_per_iter", False):
            self.policy.reset_optimizer()
        return {
            "round": k,
            "beta": self.beta(k),
            "new_samples": len(obs_r),
            "buffer_size": self.buffer_size(),
            # train_loss in the policy's training-loss space (NLL / MSE /
            # PPO-clip); val_mse always MSE-on-mean for cross-run compare.
            "train_loss": train_loss,
            "val_mse": val_mse,
        }
