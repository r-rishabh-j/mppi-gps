"""MPPI-GPS for DeterministicPolicy.

Variant of ``mppi_gps_clip.py`` for a non-stochastic student policy.

* Policy prior is a quadratic distance to the policy mean::

      prior(states, actions) = alpha * sum_t || a_t - pi(obs_t) ||^2          (K,)

  Non-negative, in env-cost units. MPPI adds it to S_k directly
  (``S = costs + is_corr + prior``); higher alpha pulls samples toward the
  current policy.

* Distillation is always MSE on (obs, action_label) pairs aggregated
  across conditions / sub-episodes / (optionally) the cross-iter buffer.

* Stabilisers retained: EMA shadow + hard-sync, Adam reset, replay buffer
  (``distill_buffer_cap``), DAgger-style relabel (gated on ``alpha > 0``),
  warm-start MPPI from the current policy, auto-reset on termination,
  gradient-norm clipping inside the policy's mse_step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.evaluation import evaluate_policy
from src.utils.experiment import copy_as, save_checkpoint


# ---------------------------------------------------------------------------
# Policy prior for MPPI (squared distance to deterministic policy mean)
# ---------------------------------------------------------------------------

def make_policy_prior(
    policy: DeterministicPolicy,
    env: BaseEnv,
    alpha: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a per-sample cost callable for ``MPPI.plan_step``.

    Given (states, actions) of shapes ``(K, H, state_dim)`` and
    ``(K, H, act_dim)``, returns::

        C_k = alpha * sum_t || a_{k,t} - pi(obs(s_{k,t})) ||^2          (K,)

    Non-negative; in the same units as env cost. MPPI adds this to ``S_k``
    inside the softmin, so larger residuals → higher cost → lower weight,
    which biases the samples toward the policy mean.

    The policy's ``forward`` is called inside ``torch.no_grad()`` and the
    caller is responsible for ensuring the policy is in ``eval()`` mode
    (Dropout / LayerNorm in ``DeterministicPolicy`` would otherwise inject
    noise into the cost).
    """
    device = policy._device

    def prior_cost(
        states: np.ndarray,
        actions: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        states = np.asarray(states)
        actions = np.asarray(actions)
        K, H, act_dim = actions.shape

        # Full physics state → policy observation.
        obs = np.asarray(env.state_to_obs(states, sensordata))  # (K, H, obs_dim)
        obs_flat = obs.reshape(K * H, -1)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_flat, dtype=torch.float32, device=device)
            mu_flat = policy.forward(obs_t).cpu().numpy()

        mu = mu_flat.reshape(K, H, act_dim)
        sq = ((actions - mu) ** 2).sum(axis=(1, 2))   # (K,)
        return alpha * sq

    return prior_cost


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@dataclass
class GPSHistory:
    """Per-iteration metrics."""
    iteration_costs: list[float] = field(default_factory=list)
    iteration_eval_costs: list[float] = field(default_factory=list)
    distill_losses: list[float] = field(default_factory=list)
    iteration_ema_drift: list[float] = field(default_factory=list)
    best_iter: int = -1
    best_cost: float = float("inf")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MPPIGPSDet:
    """MPPI-GPS with a DeterministicPolicy student. See module docstring."""

    def __init__(
        self,
        env: BaseEnv,
        mppi_cfg: MPPIConfig,
        policy_cfg: PolicyConfig,
        gps_cfg: GPSConfig,
        device: torch.device | str | None = None,
    ):
        self.env = env
        self.mppi_cfg = mppi_cfg
        self.policy_cfg = policy_cfg
        self.gps_cfg = gps_cfg

        self.policy = DeterministicPolicy(
            env.obs_dim,
            env.action_dim,
            policy_cfg,
            device=device,
            action_bounds=env.action_bounds,
        )

        self.mppi = MPPI(env, mppi_cfg)

        # Cross-iteration episode replay buffer (rows-capped, FIFO eviction by
        # whole episodes). Empty when distill_buffer_cap == 0.
        self._episode_buffer: list[dict] = []

        # EMA shadow over trainable params. Eval and best.pt selection happen
        # inside an ema_swapped_in() window, so the on-disk checkpoint matches
        # the reported eval cost.
        if gps_cfg.ema_decay > 0.0:
            self.policy.attach_ema(gps_cfg.ema_decay)

    # ----- helpers ---------------------------------------------------------

    def _sample_initial_conditions(self, n: int) -> list[np.ndarray]:
        conditions = []
        for _ in range(n):
            self.env.reset()
            conditions.append(self.env.get_state().copy())
        return conditions

    def _distill_epoch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,  # accepted for parity; unused in MSE path
    ) -> float:
        """MSE distillation with gradient-norm clipping.

        ``weights`` is currently uniform across rows (set to ones in the
        C-step) and the deterministic policy has no weighted loss, so we
        ignore it. Kept in the signature so the buffer flatten path doesn't
        need a special case.

        When ``cfg.grad_clip_norm > 0``, the per-batch parameter gradient
        is L2-norm-clipped to that bound inside ``DeterministicPolicy.
        mse_step`` (between backward and the Adam step). Bounds per-update
        parameter movement directly — loss-agnostic, dimension-aware, no
        biased estimator. Replaces the earlier action-target clip
        (``cfg.clip_eps``), which censored noisy labels and created a
        biased fixed point at ``pi_old(o) ± clip_eps``.

        ``cfg.grad_clip_norm == 0`` falls through to plain ``mse_step``.
        """
        del weights
        cfg = self.gps_cfg
        N = len(obs)
        indices = np.arange(N)
        last_loss = 0.0
        grad_clip_norm = float(getattr(cfg, "grad_clip_norm", 0.0))

        for _ in range(cfg.distill_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, cfg.distill_batch_size):
                batch = indices[start : start + cfg.distill_batch_size]
                last_loss = self.policy.mse_step(
                    obs[batch], actions[batch],
                    grad_clip_norm=grad_clip_norm,
                )
        return last_loss

    def _warm_start_mppi(self, initial_state: np.ndarray):
        """Seed MPPI nominal U with a policy rollout from ``initial_state``."""
        self.env.reset(state=initial_state)
        actions: list[np.ndarray] = []
        done = False
        for _ in range(self.mppi_cfg.H):
            if done:
                actions.append(np.zeros(self.env.action_dim))
                continue
            obs = self.env._get_obs()
            action = self.policy.act_np(obs)
            actions.append(action)
            _, _, done, _ = self.env.step(action)
        self.mppi.U = np.array(actions)

    # ----- main loop -------------------------------------------------------

    def train(
        self,
        num_iterations: int | None = None,
        run_dir: Path | None = None,
    ) -> GPSHistory:
        num_iterations = num_iterations or self.gps_cfg.num_iterations
        cfg = self.gps_cfg
        history = GPSHistory()

        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)

        outer_bar = tqdm(range(num_iterations), desc="GPS-det", unit="iter")
        for iteration in outer_bar:
            alpha = cfg.policy_augmented_alpha

            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []
            condition_costs: list[float] = []

            # ========== C-STEP ==========
            # eval() so the prior_fn forward (which goes through Dropout +
            # LayerNorm) is deterministic — Dropout active in train() mode
            # would inject noise into MPPI's cost.
            self.policy.eval()

            c_step_total = cfg.num_conditions * cfg.episode_length
            c_bar = tqdm(
                total=c_step_total,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )
            for ic_idx, ic_state in enumerate(initial_conditions):
                # New prior closure each condition — captures current policy
                # weights (for `policy.forward`) and current alpha.
                prior_fn = make_policy_prior(self.policy, self.env, alpha)

                if iteration > 0 and cfg.warm_start_policy:
                    self._warm_start_mppi(ic_state)
                else:
                    self.mppi.reset()

                self.env.reset(state=ic_state)

                # Sub-episodes split at every `done` boundary so a single
                # condition with auto_reset=True yields multiple entries.
                condition_sub_episodes: list[dict] = []
                cur_obs: list[np.ndarray] = []
                cur_actions: list[np.ndarray] = []
                episode_cost = 0.0

                def _flush_sub_episode():
                    if not cur_obs:
                        return
                    condition_sub_episodes.append({
                        "obs": np.array(cur_obs),
                        "actions": np.array(cur_actions),
                        "weights": np.ones(len(cur_obs)),
                    })
                    cur_obs.clear()
                    cur_actions.clear()

                for t in range(cfg.episode_length):
                    state = self.env.get_state()
                    obs = self.env._get_obs()

                    # Executor: MPPI with the policy prior. Steers the env.
                    action_exec, _info = self.mppi.plan_step(state, prior=prior_fn)

                    # DAgger-style relabel: a fresh prior-free MPPI call
                    # whose action is the training label. dry_run=True keeps
                    # MPPI's persistent state intact so the executor is
                    # undisturbed. Gated on alpha > 0 (no point relabeling
                    # when the prior is already inactive).
                    if cfg.dagger_relabel and alpha > 0.0:
                        action_label, _ = self.mppi.plan_step(
                            state, prior=None, dry_run=True,
                        )
                    else:
                        action_label = action_exec

                    cur_obs.append(obs.copy())
                    cur_actions.append(action_label.copy())

                    _, cost, done, _ = self.env.step(action_exec)
                    episode_cost += cost
                    c_bar.update(1)

                    if done and t < cfg.episode_length - 1:
                        _flush_sub_episode()
                        if cfg.auto_reset:
                            self.env.reset()
                            self.mppi.reset()
                            continue
                        c_bar.update(cfg.episode_length - (t + 1))
                        break

                _flush_sub_episode()

                # Route into either the cross-iter buffer or per-iter lists.
                if cfg.distill_buffer_cap > 0:
                    for sub_ep in condition_sub_episodes:
                        self._episode_buffer.append(sub_ep)
                    total_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                    while (total_rows > cfg.distill_buffer_cap
                           and len(self._episode_buffer) > 1):
                        dropped = self._episode_buffer.pop(0)
                        total_rows -= len(dropped["obs"])
                else:
                    for sub_ep in condition_sub_episodes:
                        all_obs.append(sub_ep["obs"])
                        all_actions.append(sub_ep["actions"])
                        all_weights.append(sub_ep["weights"])

                condition_costs.append(episode_cost)
                c_bar.set_postfix(
                    ic=f"{ic_idx + 1}/{cfg.num_conditions}",
                    last_cost=f"{episode_cost:.1f}",
                )
            c_bar.close()

            # ========== S-STEP ==========
            # train() so Dropout + LayerNorm-running-stats behave correctly
            # for gradient steps.
            self.policy.train()

            if cfg.distill_buffer_cap > 0:
                obs_flat = np.concatenate(
                    [ep["obs"] for ep in self._episode_buffer], axis=0)
                act_flat = np.concatenate(
                    [ep["actions"] for ep in self._episode_buffer], axis=0)
                w_flat = np.concatenate(
                    [ep["weights"] for ep in self._episode_buffer], axis=0)
            else:
                obs_flat = np.concatenate(all_obs, axis=0)
                act_flat = np.concatenate(all_actions, axis=0)
                w_flat = np.concatenate(all_weights, axis=0)

            loss = self._distill_epoch(obs_flat, act_flat, w_flat)

            # Diagnostic: post-distill EMA drift. Compute BEFORE any hard-sync
            # so the diagnostic reflects how far theta actually moved this
            # S-step (post-sync drift would always be 0).
            ema_drift = self.policy.ema_l2_drift() if cfg.ema_decay > 0.0 else float("nan")

            # End-of-S-step stabilisers: hard-sync first, then Adam reset.
            if cfg.ema_decay > 0.0 and cfg.ema_hard_sync:
                self.policy.ema_sync()
            if cfg.reset_optim_per_iter:
                self.policy.reset_optimizer()

            # ========== Logging ==========
            mppi_cost = float(np.mean(condition_costs))
            history.iteration_costs.append(mppi_cost)
            history.distill_losses.append(loss)
            history.iteration_ema_drift.append(ema_drift)

            # ========== Eval (drives best.pt) ==========
            eval_every = max(int(getattr(cfg, "eval_every", 1)), 1)
            is_last = (iteration == num_iterations - 1)
            do_eval = is_last or ((iteration + 1) % eval_every == 0)

            eval_cost = float("nan")
            with self.policy.ema_swapped_in():
                if do_eval:
                    self.policy.eval()
                    eval_stats = evaluate_policy(
                        self.policy, self.env,
                        n_episodes=cfg.n_eval_eps,
                        episode_len=cfg.eval_ep_len,
                        seed=iteration,
                    )
                    eval_cost = float(eval_stats["mean_cost"])
                history.iteration_eval_costs.append(eval_cost)

                if run_dir is not None:
                    iter_path = run_dir / f"iter_{iteration:03d}.pt"
                    save_checkpoint(
                        iter_path, self.policy,
                        iteration=iteration,
                        mppi_cost=mppi_cost,
                        eval_cost=eval_cost,
                        distill_loss=loss,
                        ema_drift=ema_drift,
                    )
                    if do_eval and eval_cost < history.best_cost:
                        history.best_cost = eval_cost
                        history.best_iter = iteration
                        copy_as(iter_path, run_dir / "best.pt")

            postfix = {
                "mppi_cost": f"{mppi_cost:.2f}",
                "loss": f"{loss:.3f}",
                "best": history.best_iter if history.best_iter >= 0 else "-",
            }
            if do_eval:
                postfix["eval_cost"] = f"{eval_cost:.2f}"
            outer_bar.set_postfix(**postfix)

            base_line = (
                f"[GPS-det iter {iteration:3d}]  "
                f"mppi_cost={mppi_cost:8.2f}  "
                f"distill_loss={loss:.4f}"
            )
            if cfg.distill_buffer_cap > 0:
                n_eps = len(self._episode_buffer)
                n_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                base_line += f"  buf={n_eps}eps/{n_rows}rows"
            if cfg.ema_decay > 0.0:
                base_line += f"  ema_drift={ema_drift:.4f}"
            if do_eval:
                base_line += f"  eval_cost={eval_cost:8.2f}"
            tqdm.write(base_line)

        outer_bar.close()
        return history
