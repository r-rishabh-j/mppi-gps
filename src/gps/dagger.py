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

        # growing buffer — flat rows of (obs, a*)
        self._buf_obs: list[np.ndarray] = []
        self._buf_act: list[np.ndarray] = []
        self._buf_round: list[np.ndarray] = []

        # Optional EMA of policy weights. DAgger trains MSE-on-mean, so only
        # the EMA stabiliser applies (no KL-to-prev: there's no policy
        # distribution to constrain in the deterministic case).
        if cfg.ema_decay > 0.0:
            self.policy.attach_ema(cfg.ema_decay)

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

    def _collect(
        self,
        n_rollouts: int,
        beta: float,
        seed_base: int,
        episode_len: int | None = None,
        progress_label: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out `n_rollouts` episodes under the given β and return flat
        (obs, expert_actions). Always queries MPPI for the relabel target."""
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
                        # auto_reset → keep collecting until ep_len so the
                        # progress bar / per-episode budget aren't wasted on
                        # an early termination (matters for terminating envs
                        # like hopper). Otherwise break — terminating mid-
                        # episode produces a shorter trajectory which is the
                        # honest signal for non-auto-reset envs.
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
        """Roll out N_roll trajectories with β-mixing and relabel every state."""
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
        """Pre-train the policy on pure-MPPI rollouts before the DAgger loop.

        Appends to the aggregate buffer with round_idx=-1 so warmup data is
        reused across subsequent finetune() calls. Returns per-epoch train MSE.

        If `cache_path` is given:
          - and the file exists: load flat (states, actions) from it and skip
            the (expensive) MPPI rollout collection. Trajectory-shaped arrays
            from a BC-style h5 (M, T, dim) are auto-flattened, so a
            `collect_bc_demos.py` dataset works here too.
          - otherwise: collect fresh, then save to `cache_path` using the same
            schema as `collect_bc_demos.py` so the file is reusable both ways.
        If `cache_path` is None, nothing is read or written — this is the
        default; pass a path only when you explicitly want to persist.

        The BC training epochs always run — we only cache the collection step,
        which is where the wall-clock cost lives. To invalidate a stale cache,
        delete the file or pass a new path.
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
            # BC-style caches are (M, T, dim); warmup-native caches are (N, dim).
            # Flatten either to (N, dim) so the buffer gets consistent rows.
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
                seed_base=self.cfg.seed + 999_000,  # distinct from any round seed
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
        # Warmup never runs the Gaussian PPO clip — random-init policy gives
        # a meaningless ratio. Deterministic grad-norm clip is fine here.
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
            epoch_bar.set_postfix(train_mse=f"{loss:.4f}")
        return losses

    # ---------- training ----------
    def _train_step(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        grad_clip_norm: float = 0.0,
        old_policy: "GaussianPolicy | None" = None,
        clip_ratio: float = 0.2,
    ) -> float:
        """One distillation step. Two policy-class-specific paths:

        * Deterministic: MSE on the policy mean. ``grad_clip_norm > 0``
          activates an L2 gradient-norm clip inside ``DeterministicPolicy.
          mse_step`` (between backward and the Adam step).
        * Gaussian + ``old_policy`` + ``clip_ratio > 0``: PPO-style ratio
          clip surrogate, mirroring ``mppi_gps_clip._train_step_clipped``.
          With uniform "advantage" of 1 (DAgger has no MPPI weights at
          training time), each row pulls its log-likelihood up to
          ``ratio = 1 + clip_ratio`` per step, then saturates.
        * Gaussian without those args: plain MSE-on-mean, the historical
          DAgger default.
        """
        # Deterministic path -------------------------------------------------
        if isinstance(self.policy, DeterministicPolicy):
            if grad_clip_norm > 0.0:
                return self.policy.mse_step(obs, act, grad_clip_norm=grad_clip_norm)
            return self.policy.mse_step(obs, act)

        # Gaussian + PPO clip ------------------------------------------------
        if (
            isinstance(self.policy, GaussianPolicy)
            and old_policy is not None
            and clip_ratio > 0.0
        ):
            return self._train_step_gaussian_ppo(obs, act, old_policy, clip_ratio)

        # Gaussian default: plain MSE on mean --------------------------------
        return self.policy.mse_step(obs, act)

    def _train_step_gaussian_ppo(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        old_policy: GaussianPolicy,
        clip_ratio: float,
    ) -> float:
        """PPO-style ratio-clipped surrogate for the Gaussian dagger fit.

        ``L = - E[ min(r, clip(r, 1-eps, 1+eps)) ]`` where
        ``r = pi_theta(a|o) / pi_old(a|o)``. Mirrors the Gaussian branch of
        ``mppi_gps_clip._train_step_clipped`` but with constant advantage 1
        (DAgger does not retain MPPI importance weights past collection).

        Touches the same per-step machinery as ``GaussianPolicy.mse_step``:
        normaliser update, optimizer.zero_grad/backward/step, EMA update.
        """
        device = self.policy.device
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(act, dtype=torch.float32, device=device)

        if self.policy.normalizer is not None:
            self.policy.normalizer.update(o_t)

        curr_log_prob = self.policy.log_prob(o_t, a_t)
        with torch.no_grad():
            old_log_prob = old_policy.log_prob(o_t, a_t)

        ratio = torch.exp(curr_log_prob - old_log_prob)
        # Uniform "advantage" = 1; min(...) caps boost at ratio = 1+clip_ratio.
        surr1 = ratio
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        loss = -torch.min(surr1, surr2).mean()

        self.policy.optimizer.zero_grad()
        loss.backward()
        # NaN-loss guard: same rationale as in GaussianPolicy.train_weighted.
        if not torch.isfinite(loss):
            self.policy.optimizer.zero_grad()
            return float("nan")
        self.policy.optimizer.step()
        # EMA hook — `mse_step` does this; mirror so the EMA shadow stays
        # in sync with the live policy across PPO updates too. (This is
        # one consistency improvement over `mppi_gps_clip`'s Gaussian
        # branch, which omits the EMA update.)
        if self.policy.ema is not None:
            self.policy.ema.update(self.policy)
        return float(loss.item())

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

    def finetune(
        self,
        latest_obs: np.ndarray,
        latest_act: np.ndarray,
        round_idx: int | None = None,
    ) -> tuple[float, float]:
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
        grad_clip_norm = float(getattr(self.cfg, "grad_clip_norm", 0.0))
        clip_ratio = float(getattr(self.cfg, "clip_ratio", 0.0))

        # Pre-finetune snapshot for the Gaussian PPO clip. Frozen, eval mode,
        # no grad. Skipped for deterministic policy or when clip_ratio == 0
        # (avoid the deepcopy + per-batch forward when not in use).
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
            epoch_bar.set_postfix(train_mse=f"{last_train:.4f}", val_mse=f"{last_val:.4f}")
        return last_train, last_val

    # ---------- full iteration ----------
    def step(self, k: int) -> dict:
        obs_r, act_r = self.collect_round(k)
        self.append(obs_r, act_r, round_idx=k)
        train_mse, val_mse = self.finetune(obs_r, act_r, round_idx=k)
        # End-of-round stabilisers, mirroring MPPIGPS:
        #  - ema_hard_sync: θ ← EMA so next round's rollouts run the smoothed policy.
        #  - reset_optim_per_iter: wipe Adam moments so stale momentum doesn't
        #    carry into the next round (especially important after a hard-sync).
        if getattr(self.cfg, "ema_decay", 0.0) > 0.0 and getattr(self.cfg, "ema_hard_sync", False):
            self.policy.ema_sync()
        if getattr(self.cfg, "reset_optim_per_iter", False):
            self.policy.reset_optimizer()
        return {
            "round": k,
            "beta": self.beta(k),
            "new_samples": len(obs_r),
            "buffer_size": self.buffer_size(),
            "train_mse": train_mse,
            "val_mse": val_mse,
        }
