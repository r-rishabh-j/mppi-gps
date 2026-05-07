"""Warp-batched MPPI-GPS trainer.

Drop-in replacement for ``MPPIGPS`` when the env is a Warp variant
(``AdroitRelocateWarp``) and the user wants the GPS C-step's
``num_conditions`` loop collapsed into one parallel timestep advance.

Architecture (vs. CPU MPPIGPS):

    CPU path: for each condition n in [0..N):
                  for t in [0..T): plan_step(state[n]) → step env
              N × T × MPPI calls per iter; conditions sequential.

    Warp path: for t in [0..T):
                   plan_step(states[N]) → actions[N, nu]
                   # one warp graph launch over N*K worlds
                   step each of N CPU envs in parallel
               T × MPPI calls per iter; conditions parallel.

Wallclock win on Adroit: at typical settings (N=5 conditions, T=500
steps, K=1024 samples, H=64), the CPU path makes 2,500 MPPI calls per
iter, each rolling 1,024 contact-rich Adroit sims through a CPU thread
pool. The warp path makes 500 MPPI calls, each rolling 5,120 sims on
GPU through a captured CUDA graph — usually 5-15× faster end-to-end
depending on the GPU.

----------------------------------------------------------------------
Scope cuts vs. ``MPPIGPS``
----------------------------------------------------------------------
- ``warm_start_policy`` not honored (would need per-condition policy
  rollouts; defer until measured-need).
- ``dagger_relabel`` not honored (would need a second batched plan call
  with no prior — easy to add later, ~10 lines).
- ``auto_reset`` ignored (Adroit relocate doesn't terminate mid-episode).
- α=0 h5 cache disabled — duplicates standalone collect_bc_demos.
- ``open_loop_steps > 1`` ignored (BatchedMPPI forces 1).
- ``adaptive_lam`` ignored (BatchedMPPI uses fixed λ).

S-step (distillation) and EMA / KL-adaptive / eval are inherited from
``MPPIGPS`` unchanged — they operate on collected (obs, action) pairs,
which look identical to the CPU path's output.
"""
from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.gps.mppi_gps_unified import (
    GPSHistory,
    MPPIGPS,
    estimate_kl_p_to_policy,
    make_policy_prior,
    schedule_alpha,
    update_alpha_kl_adaptive,
)
from src.mppi.batched_mppi import BatchedMPPI
from src.utils.config import GPSConfig, MPPIConfig, PolicyConfig
from src.utils.evaluation import evaluate_policy
from src.utils.experiment import copy_as, save_checkpoint


def _step_one_data(
    env: BaseEnv,
    data: mujoco.MjData,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Advance ONE MjData instance by ``env._frame_skip`` mj_step calls.

    Generic version of ``MuJoCoEnv.step`` that operates on a passed-in
    ``data`` instead of ``env.data`` so we can run N independent
    simulations from a single env class. Matches the CPU step's output
    shape conventions (state in FULLPHYSICS layout, sensor flat copy,
    cost from the env's running_cost on (1,1,*) inputs).
    """
    data.ctrl[:] = action
    for _ in range(env._frame_skip):
        mujoco.mj_step(env.model, data)
    state = np.empty(env._nstate)
    mujoco.mj_getState(
        env.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    sensor = data.sensordata.copy()
    cost = env.running_cost(
        state.reshape(1, 1, -1),
        action.reshape(1, 1, -1),
        sensor.reshape(1, 1, -1),
    ).item()
    return state, sensor, float(cost)


def _set_state_one_data(
    env: BaseEnv,
    data: mujoco.MjData,
    state: np.ndarray,
) -> None:
    """In-place ``mj_setState`` + ``mj_forward`` on a passed-in MjData."""
    mujoco.mj_setState(
        env.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    mujoco.mj_forward(env.model, data)


class WarpMPPIGPS(MPPIGPS):
    """GPS trainer that batches the C-step over ``num_conditions`` envs.

    The env passed in must be a Warp variant (``env._use_warp == True``)
    constructed with ``nworld = num_conditions × cfg.K``. This class
    creates ``num_conditions`` CPU ``MjData`` twins from ``env.model``
    (one per condition's executor) and uses ``BatchedMPPI`` for all
    MPPI rollouts.

    Inherits S-step (`_distill_epoch`), eval, history, EMA stabilisers
    from ``MPPIGPS`` unchanged.
    """

    def __init__(
        self,
        env: BaseEnv,
        mppi_cfg: MPPIConfig,
        policy_cfg: PolicyConfig,
        gps_cfg: GPSConfig,
        device: torch.device | str | None = None,
        deterministic: bool = False,
    ):
        if not getattr(env, "_use_warp", False):
            raise ValueError(
                "WarpMPPIGPS requires a Warp-backed env "
                "(env._use_warp must be True). Use the CPU MPPIGPS otherwise."
            )

        # ---- Reproduce MPPIGPS.__init__ minus the `self.mppi = MPPI(...)` line ----
        # We don't use plain MPPI; we use BatchedMPPI. Everything else
        # (policy, prior_type resolution, ema attach, replay buffer slot,
        # KL-adaptive lazy field) is identical to the parent.
        self.env = env
        self.mppi_cfg = mppi_cfg
        self.policy_cfg = policy_cfg
        self.gps_cfg = gps_cfg
        self._deterministic = deterministic

        # Policy
        if deterministic:
            from src.policy.deterministic_policy import DeterministicPolicy
            self.policy = DeterministicPolicy(
                env.obs_dim, env.action_dim, policy_cfg,
                device=device, action_bounds=env.action_bounds,
            )
        else:
            from src.policy.gaussian_policy import GaussianPolicy
            self.policy = GaussianPolicy(
                env.obs_dim, env.action_dim, policy_cfg,
                device=device, action_bounds=env.action_bounds,
            )

        # Prior type (same auto-resolve as MPPIGPS)
        prior_type = getattr(gps_cfg, "policy_prior_type", "auto")
        if prior_type == "auto":
            prior_type = "mean_distance" if deterministic else "nll"
        if prior_type == "nll" and deterministic:
            raise ValueError(
                "policy_prior_type='nll' requires GaussianPolicy."
            )
        if prior_type not in ("nll", "mean_distance"):
            raise ValueError(
                f"unknown policy_prior_type: {prior_type!r}"
            )
        self._prior_type = prior_type

        # ---- BatchedMPPI ----
        self.mppi = BatchedMPPI(
            env, mppi_cfg, num_conditions=gps_cfg.num_conditions,
        )
        self.N = gps_cfg.num_conditions

        # ---- N CPU MjData twins for executor stepping ----
        # Each is a full MjData allocation; size is dominated by Adroit's
        # contact buffer (~MB per condition, fine for typical N≤16).
        self._batch_data: list[mujoco.MjData] = [
            mujoco.MjData(env.model) for _ in range(self.N)
        ]

        # Inherited bookkeeping
        self._episode_buffer: list[dict] = []
        self._kl_alpha: float | None = None
        if gps_cfg.ema_decay > 0.0:
            self.policy.attach_ema(gps_cfg.ema_decay)

        # Loud opt-out warnings for unsupported features.
        import warnings
        for attr in ("warm_start_policy", "dagger_relabel", "auto_reset"):
            if getattr(gps_cfg, attr, False):
                warnings.warn(
                    f"WarpMPPIGPS does not honor GPSConfig.{attr}=True; "
                    "feature is ignored on the warp path. Use the CPU "
                    "trainer if you need it.",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_all_to(self, initial_states: list[np.ndarray]) -> None:
        """Set each of the N executor MjData twins to its condition's state."""
        assert len(initial_states) == self.N
        for n in range(self.N):
            _set_state_one_data(self.env, self._batch_data[n], initial_states[n])

    def _get_states_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """Snapshot per-condition (state, obs) from the N executor MjDatas.

        Returns:
            states: (N, nstate) FULLPHYSICS layout.
            obs:    (N, obs_dim) — obs computed via env.state_to_obs (uses
                    sensordata where available so DAPG-obs envs hit the
                    fast path).
        """
        states = np.stack([
            self._mj_getstate(self._batch_data[n]) for n in range(self.N)
        ])
        sensors = np.stack([
            self._batch_data[n].sensordata.copy() for n in range(self.N)
        ])
        # state_to_obs expects (..., nstate); pass (1, N, ...) so envs
        # that index along last axis work, then squeeze leading 1.
        obs = np.asarray(
            self.env.state_to_obs(states[None], sensors[None])
        )[0]
        return states, obs

    def _mj_getstate(self, data: mujoco.MjData) -> np.ndarray:
        """Like MuJoCoEnv.get_state but for an arbitrary MjData."""
        state = np.empty(self.env._nstate)
        mujoco.mj_getState(
            self.env.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        return state

    # ------------------------------------------------------------------
    # train()
    # ------------------------------------------------------------------

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
        kl_adaptive_enabled = (
            cfg.kl_target > 0.0 and not self._deterministic
        )

        outer_bar = tqdm(range(num_iterations), desc="GPS-warp", unit="iter")
        for iteration in outer_bar:
            # ---- α schedule / KL-adaptive seed (same as MPPIGPS) ----
            use_kl_adaptive_this_iter = (
                kl_adaptive_enabled and iteration >= cfg.alpha_warmup_iters
            )
            if use_kl_adaptive_this_iter:
                if self._kl_alpha is None:
                    self._kl_alpha = schedule_alpha(iteration, cfg)
                alpha = self._kl_alpha
            else:
                alpha = schedule_alpha(iteration, cfg)

            # ---- Reset all N executors to initial conditions ----
            self._reset_all_to(initial_conditions)
            self.mppi.reset()

            # ---- Per-condition trajectory accumulators ----
            ep_obs = [[] for _ in range(self.N)]
            ep_actions = [[] for _ in range(self.N)]
            ep_costs = np.zeros(self.N)

            # ---- KL-adaptive accumulator (per-condition, per-step) ----
            kl_obs_buf: list[np.ndarray] = []
            kl_mu_p_buf: list[np.ndarray] = []
            kl_var_p_buf: list[np.ndarray] = []

            # ---- Build prior closure (same factory as CPU path) ----
            self.policy.eval()
            if alpha > 0.0:
                prior_fn = make_policy_prior(
                    self.policy, self.env, alpha, self._prior_type,
                )
            else:
                prior_fn = None

            # ========== BATCHED C-STEP ==========
            c_bar = tqdm(
                total=cfg.episode_length,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )
            for t in range(cfg.episode_length):
                states, obs = self._get_states_obs()           # (N, *)

                # Single batched plan: K rollouts × N conditions in one
                # graph launch.
                actions, info = self.mppi.plan_step(states, prior=prior_fn)
                # actions: (N, nu)

                # KL-adaptive: per-condition (μ_p, σ_p²) from the unbiased
                # weights over K samples. Pushed into the buffer once per
                # (n, t).
                if use_kl_adaptive_this_iter:
                    ws = self.mppi._last_unbiased_weights              # (N, K)
                    # _last_actions[n, k, h, nu]; first-step is h=0.
                    first_actions = self.mppi._last_actions[:, :, 0, :] # (N, K, nu)
                    mu_p = (ws[..., None] * first_actions).sum(axis=1) # (N, nu)
                    var_p = (
                        ws[..., None] * (first_actions - mu_p[:, None, :]) ** 2
                    ).sum(axis=1)                                      # (N, nu)
                    for n in range(self.N):
                        kl_obs_buf.append(obs[n].copy())
                        kl_mu_p_buf.append(mu_p[n])
                        kl_var_p_buf.append(var_p[n])

                # Advance each executor MjData by one control step.
                # Sequential CPU loop — N is typically small (5-10) and
                # each step is just `frame_skip` mj_step calls; batching
                # this further would mean another mj_warp instance for
                # the executor. Defer until profiled.
                for n in range(self.N):
                    state_n, sensor_n, cost_n = _step_one_data(
                        self.env, self._batch_data[n], actions[n],
                    )
                    ep_obs[n].append(obs[n].copy())
                    ep_actions[n].append(actions[n].copy())
                    ep_costs[n] += cost_n

                c_bar.update(1)
            c_bar.close()

            # ---- Aggregate sub-episodes (one per condition; no auto_reset) ----
            condition_sub_episodes = [
                {
                    "obs": np.array(ep_obs[n]),
                    "actions": np.array(ep_actions[n]),
                    "weights": np.ones(len(ep_obs[n])),
                }
                for n in range(self.N) if ep_obs[n]
            ]

            if cfg.distill_buffer_cap > 0:
                self._episode_buffer.extend(condition_sub_episodes)
                total_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                while (total_rows > cfg.distill_buffer_cap
                       and len(self._episode_buffer) > 1):
                    dropped = self._episode_buffer.pop(0)
                    total_rows -= len(dropped["obs"])
                obs_flat = np.concatenate(
                    [ep["obs"] for ep in self._episode_buffer], axis=0)
                act_flat = np.concatenate(
                    [ep["actions"] for ep in self._episode_buffer], axis=0)
                w_flat = np.concatenate(
                    [ep["weights"] for ep in self._episode_buffer], axis=0)
            else:
                obs_flat = np.concatenate(
                    [ep["obs"] for ep in condition_sub_episodes], axis=0)
                act_flat = np.concatenate(
                    [ep["actions"] for ep in condition_sub_episodes], axis=0)
                w_flat = np.concatenate(
                    [ep["weights"] for ep in condition_sub_episodes], axis=0)

            # ========== S-STEP (inherited from MPPIGPS) ==========
            self.policy.train()

            prev_policy = None
            if (
                not self._deterministic
                and cfg.prev_iter_kl_coef > 0.0
            ):
                prev_policy = copy.deepcopy(self.policy)
                prev_policy.eval()
                for p in prev_policy.parameters():
                    p.requires_grad_(False)

            loss = self._distill_epoch(
                obs_flat, act_flat, w_flat, prev_policy=prev_policy,
            )

            ema_drift = (
                self.policy.ema_l2_drift() if cfg.ema_decay > 0.0 else float("nan")
            )

            if (
                prev_policy is not None
                and not self._deterministic
                and len(obs_flat) > 0
                and hasattr(self.policy, "kl_to_np")
            ):
                diag_n = min(len(obs_flat), 4096)
                diag_idx = np.random.choice(len(obs_flat), size=diag_n, replace=False)
                prev_iter_kl_diag = self.policy.kl_to_np(
                    obs_flat[diag_idx], prev_policy,
                )
            else:
                prev_iter_kl_diag = float("nan")

            if cfg.ema_decay > 0.0 and cfg.ema_hard_sync:
                self.policy.ema_sync()
            if cfg.reset_optim_per_iter:
                self.policy.reset_optimizer()

            # ========== KL-adaptive α update ==========
            kl_est_iter = float("nan")
            if use_kl_adaptive_this_iter and kl_obs_buf:
                if cfg.kl_sigma_floor_frac > 0.0:
                    if self.policy._has_act_norm:
                        scale_np = self.policy._act_scale.detach().cpu().numpy()
                        sigma_p_floor = (
                            self.mppi.sigma * cfg.kl_sigma_floor_frac / scale_np
                        )
                    else:
                        sigma_p_floor = self.mppi.sigma * cfg.kl_sigma_floor_frac
                else:
                    sigma_p_floor = 1e-6
                with self.policy.ema_swapped_in():
                    self.policy.eval()
                    kl_est_iter = estimate_kl_p_to_policy(
                        np.stack(kl_obs_buf),
                        np.stack(kl_mu_p_buf),
                        np.stack(kl_var_p_buf),
                        self.policy,
                        sigma_p_floor_norm=sigma_p_floor,
                    )
                self._kl_alpha = update_alpha_kl_adaptive(alpha, kl_est_iter, cfg)

            # ========== Logging ==========
            mppi_cost = float(np.mean(ep_costs))
            history.iteration_costs.append(mppi_cost)
            history.distill_losses.append(loss)
            history.iteration_ema_drift.append(ema_drift)
            history.iteration_prev_iter_kl.append(prev_iter_kl_diag)
            history.iteration_alpha.append(float(alpha))
            history.iteration_kl_est.append(kl_est_iter)

            # ========== Eval ==========
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
                        prev_iter_kl=prev_iter_kl_diag,
                    )
                    if do_eval and eval_cost < history.best_cost:
                        history.best_cost = eval_cost
                        history.best_iter = iteration
                        copy_as(iter_path, run_dir / "best.pt")

            postfix = {
                "mppi": f"{mppi_cost:.2f}",
                "loss": f"{loss:.3f}",
                "alpha": f"{alpha:.4f}",
            }
            if use_kl_adaptive_this_iter:
                postfix["kl"] = f"{kl_est_iter:.3f}"
            outer_bar.set_postfix(**postfix)

            line = (
                f"[GPS-warp iter {iteration:3d}]  alpha={alpha:.4f}  "
                f"mppi_cost={mppi_cost:8.2f}  distill_loss={loss:.4f}"
            )
            if use_kl_adaptive_this_iter:
                line += f"  kl_est={kl_est_iter:7.4f}/tgt={cfg.kl_target:.3f}"
            line += f"  eval_cost={eval_cost:8.2f}"
            print(line)

        outer_bar.close()
        if run_dir is not None:
            final_path = run_dir / "final.pt"
            save_checkpoint(final_path, self.policy)
        return history
