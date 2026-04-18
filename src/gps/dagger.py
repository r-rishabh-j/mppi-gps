"""DAgger-style distillation of MPPI into a GaussianPolicy.

Per iteration:
  1. Roll out the current policy (with β probability of executing MPPI instead).
  2. At every visited full state, query MPPI as the expert and record (obs, a*).
  3. Append to an aggregating buffer.
  4. Finetune the policy on the buffer with MSE-on-mean.

MPPI runs on CPU (numpy/MuJoCo). Policy training lives on whatever device
the policy was constructed with (cpu/mps/cuda). The numpy↔torch boundary
is kept at the trainer methods so callers don't have to think about it.
"""
from __future__ import annotations

import numpy as np
import torch

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
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

        # growing buffer — flat rows of (obs, a*)
        self._buf_obs: list[np.ndarray] = []
        self._buf_act: list[np.ndarray] = []
        self._buf_round: list[np.ndarray] = []

    # ---------- buffer ----------
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
        """Optional: warm-start the buffer from an existing BC dataset."""
        import h5py

        with h5py.File(path, "r") as f:
            s = f["states"][:].astype(np.float32)  # (M, T, obs_dim)
            a = f["actions"][:].astype(np.float32)  # (M, T, act_dim)
        self.append(s.reshape(-1, s.shape[-1]), a.reshape(-1, a.shape[-1]), round_idx=-1)

    # ---------- rollout + relabel ----------
    def beta(self, k: int) -> float:
        K = self.cfg.dagger_iters
        if self.cfg.beta_schedule == "constant_zero":
            return 0.0
        # linear: 1.0 at k=0, 0.0 at k>=K/2
        half = max(1, K // 2)
        return max(0.0, 1.0 - k / half)

    def _collect(self, n_rollouts: int, beta: float, seed_base: int,
                 episode_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Roll out `n_rollouts` episodes under the given β and return flat
        (obs, expert_actions). Always queries MPPI for the relabel target."""
        ep_len = episode_len if episode_len is not None else self.cfg.episode_len
        obs_rows: list[np.ndarray] = []
        act_rows: list[np.ndarray] = []

        for ep in range(n_rollouts):
            np.random.seed(seed_base + ep)
            self.env.reset()
            self.mppi.reset()

            for _ in range(ep_len):
                obs = self.env._get_obs()
                state = self.env.get_state()
                expert_action, _ = self.mppi.plan_step(state)

                if self.rng.random() < beta:
                    exec_action = expert_action
                else:
                    exec_action = self.policy.act_np(obs)

                obs_rows.append(obs.astype(np.float32))
                act_rows.append(expert_action.astype(np.float32))

                _, _, done, _ = self.env.step(exec_action)
                if done:
                    break

        obs_arr = np.stack(obs_rows, axis=0)
        act_arr = np.stack(act_rows, axis=0)
        if act_arr.ndim == 1:
            act_arr = act_arr[:, None]
        return obs_arr, act_arr

    def collect_round(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Roll out N_roll trajectories with β-mixing and relabel every state."""
        return self._collect(
            n_rollouts=self.cfg.rollouts_per_iter,
            beta=self.beta(k),
            seed_base=self.cfg.seed + 1000 * k,
        )

    def warmup(self, n_rollouts: int, epochs: int) -> list[float]:
        """Pre-train the policy on pure-MPPI rollouts before the DAgger loop.

        Appends to the aggregate buffer with round_idx=-1 so warmup data is
        reused across subsequent finetune() calls. Returns per-epoch train MSE.
        """
        if n_rollouts <= 0 or epochs <= 0:
            return []
        obs_r, act_r = self._collect(
            n_rollouts=n_rollouts, beta=1.0,
            seed_base=self.cfg.seed + 999_000,  # distinct from any round seed
        )
        self.append(obs_r, act_r, round_idx=-1)

        N = len(obs_r)
        idx = np.arange(N)
        losses: list[float] = []
        for _ in range(epochs):
            self.rng.shuffle(idx)
            running, nb = 0.0, 0
            for start in range(0, N, self.cfg.batch_size):
                b = idx[start:start + self.cfg.batch_size]
                running += self._train_step_mse(obs_r[b], act_r[b])
                nb += 1
            losses.append(running / max(nb, 1))
        return losses

    # ---------- training ----------
    def _train_step_mse(self, obs: np.ndarray, act: np.ndarray) -> float:
        return self.policy.mse_step(obs, act)

    @torch.no_grad()
    def _eval_mse(self, obs: np.ndarray, act: np.ndarray, batch: int = 16384) -> float:
        device = self.policy.device
        total, n = 0.0, 0
        for s in range(0, len(obs), batch):
            o = torch.as_tensor(obs[s:s + batch], dtype=torch.float32, device=device)
            a = torch.as_tensor(act[s:s + batch], dtype=torch.float32, device=device)
            pred = self.policy.action(o)
            total += float(((pred - a) ** 2).sum().item())
            n += a.numel()
        return total / max(n, 1)

    def finetune(self, latest_obs: np.ndarray, latest_act: np.ndarray) -> tuple[float, float]:
        """Finetune on the full aggregated buffer. Validation uses a held-out
        slice of the most-recent round to catch covariate shift."""
        # held-out val from most-recent round
        n_latest = len(latest_obs)
        perm = self.rng.permutation(n_latest)
        n_val = max(1, int(n_latest * self.cfg.val_frac))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        va_o, va_a = latest_obs[val_idx], latest_act[val_idx]

        # full training set: buffer minus the val slice from latest round
        all_o, all_a, _ = self.flat_buffer()
        # latest round is the last appended chunk; compute its bounds
        n_before_latest = self.buffer_size() - n_latest
        latest_train_o = latest_obs[train_idx]
        latest_train_a = latest_act[train_idx]
        tr_o = np.concatenate([all_o[:n_before_latest], latest_train_o], axis=0)
        tr_a = np.concatenate([all_a[:n_before_latest], latest_train_a], axis=0)

        N = len(tr_o)
        idx = np.arange(N)
        last_train = last_val = float("nan")
        for _ in range(self.cfg.distill_epochs):
            self.rng.shuffle(idx)
            running, nb = 0.0, 0
            for start in range(0, N, self.cfg.batch_size):
                b = idx[start:start + self.cfg.batch_size]
                running += self._train_step_mse(tr_o[b], tr_a[b])
                nb += 1
            last_train = running / max(nb, 1)
            last_val = self._eval_mse(va_o, va_a)
        return last_train, last_val

    # ---------- full iteration ----------
    def step(self, k: int) -> dict:
        obs_r, act_r = self.collect_round(k)
        self.append(obs_r, act_r, round_idx=k)
        train_mse, val_mse = self.finetune(obs_r, act_r)
        return {
            "round": k,
            "beta": self.beta(k),
            "new_samples": len(obs_r),
            "buffer_size": self.buffer_size(),
            "train_mse": train_mse,
            "val_mse": val_mse,
        }
