"""MPPI-based Guided Policy Search (policy-prior-only variant).

Loop: (C-step) run MPPI biased by ``-α·log π`` on each initial condition;
(S-step) BC-distill collected (obs, action) pairs into the policy;
optionally warm-start MPPI's nominal from the updated policy.
"""

import copy
from pathlib import Path

import numpy as np
import torch
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from src.envs.base import BaseEnv
from src.mppi.mppi import MPPI
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import MPPIConfig, PolicyConfig, GPSConfig
from src.utils.evaluation import evaluate_policy
from src.utils.experiment import copy_as, save_checkpoint


def make_policy_prior(policy: GaussianPolicy, env: BaseEnv, alpha: float):
    """Build MPPI's ``prior`` callable ``(states, actions, sensordata) → (K,)``.

    Returns ``-alpha · Σ_t log π(u_t | o_t)``. MPPI adds this to the
    trajectory cost, so a negative return raises sample weight — biasing
    rollouts toward actions likely under the current policy. The effective
    contribution to ``log w_k`` is ``+(α/λ) · Σ_t log π``.
    """
    def prior_fn(states, actions, sensordata=None) -> np.ndarray:
        states = np.asarray(states)
        actions = np.asarray(actions)
        K, H, _ = states.shape

        obs = np.asarray(env.state_to_obs(states, sensordata))  # (K, H, obs_dim)
        obs_flat = obs.reshape(K * H, -1)
        act_flat = actions.reshape(K * H, -1)
        lp = policy.log_prob_np(obs_flat, act_flat).reshape(K, H).sum(axis=1)
        return -alpha * lp

    return prior_fn


@dataclass
class GPSHistory:
    """Per-iteration training metrics."""
    # C-step MPPI rollout cost under policy prior (diagnostic).
    iteration_costs: list[float] = field(default_factory=list)
    # Greedy-policy eval cost; NaN on skipped iters. Drives best.pt.
    iteration_eval_costs: list[float] = field(default_factory=list)
    distill_losses: list[float] = field(default_factory=list)
    iteration_prev_iter_kl: list[float] = field(default_factory=list)
    best_iter: int = -1
    best_cost: float = float("inf")


class MPPIGPS:
    """MPPI-GPS (policy-prior-only). C-step: MPPI biased by ``-α·log π``.
    S-step: BC distillation on collected (obs, action) pairs."""

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

        self.policy = GaussianPolicy(
            env.obs_dim,
            env.action_dim,
            policy_cfg,
            device=device,
            action_bounds=env.action_bounds,
        )
        self.mppi = MPPI(env, mppi_cfg)

        # Cross-iteration replay buffer; one sub-episode per entry (split
        # at every `done`). Unused when ``distill_buffer_cap == 0``.
        self._episode_buffer: list[dict] = []

    def _sample_initial_conditions(self, n: int) -> list[np.ndarray]:
        """Sample n random initial conditions; capture FULLPHYSICS states."""
        conditions = []
        for _ in range(n):
            self.env.reset()
            conditions.append(self.env.get_state().copy())
        return conditions

    def _distill_epoch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
        prev_policy: "GaussianPolicy | None" = None,
    ) -> float:
        """S-step: weighted NLL on (obs, action), optional KL-to-prev penalty.

            L = -Σ_i w_i log π_θ(a_i | o_i)
                + coef · KL(π_θ || π_prev)        # if prev_policy + coef > 0

        Returns the last mini-batch loss.
        """
        cfg = self.gps_cfg
        N = len(obs)
        indices = np.arange(N)
        last_loss = 0.0
        use_prev_kl = (
            prev_policy is not None
            and cfg.prev_iter_kl_coef > 0.0
        )
        for _ in range(cfg.distill_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, cfg.distill_batch_size):
                batch = indices[start : start + cfg.distill_batch_size]
                last_loss = self.policy.train_weighted(
                    obs[batch], actions[batch], weights[batch],
                    prev_policy=prev_policy if use_prev_kl else None,
                    prev_kl_coef=cfg.prev_iter_kl_coef if use_prev_kl else 0.0,
                )
        return last_loss

    def _warm_start_mppi(self, initial_state: np.ndarray):
        """Roll out the policy for H steps; use as MPPI's nominal U."""
        self.env.reset(state=initial_state)
        actions = []
        done = False
        for _ in range(self.mppi_cfg.H):
            if done:
                # Pad post-terminal slots with zeros.
                actions.append(np.zeros(self.env.action_dim))
                continue
            obs = self.env._get_obs()
            action = self.policy.act_np(obs)
            actions.append(action)
            _, _, done, _ = self.env.step(action)
        self.mppi.U = np.array(actions)

    def train(
        self,
        num_iterations: int | None = None,
        run_dir: Path | None = None,
    ) -> GPSHistory:
        """Run the GPS loop.

        ``run_dir`` (optional): write ``iter_<k>.pt`` per iter and copy the
        best-by-eval-cost to ``best.pt`` (eval cadence via
        ``gps_cfg.eval_every``).
        """
        num_iterations = num_iterations or self.gps_cfg.num_iterations
        cfg = self.gps_cfg
        history = GPSHistory()

        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

        # Fixed initial-condition set, persisted across all iters.
        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)

        outer_bar = tqdm(range(num_iterations), desc="GPS", unit="iter")
        for iteration in outer_bar:
            alpha = cfg.policy_augmented_alpha

            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []
            condition_costs: list[float] = []

            # ========== C-STEP ==========
            c_step_total = cfg.num_conditions * cfg.episode_length
            c_bar = tqdm(
                total=c_step_total,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )
            for ic_idx, ic_state in enumerate(initial_conditions):
                # Closure captures *current* policy weights.
                prior_fn = make_policy_prior(self.policy, self.env, alpha)

                if iteration > 0 and cfg.warm_start_policy:
                    self._warm_start_mppi(ic_state)
                else:
                    self.mppi.reset()

                self.env.reset(state=ic_state)

                # One sub-episode closes per `done`; sub-episodes are
                # contiguous and never span a reset.
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

                    # Executor: prior-biased plan_step (mutates self.mppi state).
                    action_exec, info = self.mppi.plan_step(state, prior=prior_fn)

                    # DAgger-style relabel: fresh prior-free query (dry_run
                    # so the nominal U is undisturbed). Env still steps on
                    # action_exec so the visited state distribution is
                    # prior-biased; only the *label* is uncontaminated.
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
                            # Terminating env: re-seed and keep collecting.
                            self.env.reset()
                            self.mppi.reset()
                            continue
                        # No auto_reset: cut short, pad progress bar.
                        c_bar.update(cfg.episode_length - (t + 1))
                        break

                _flush_sub_episode()

                if cfg.distill_buffer_cap > 0:
                    for sub_ep in condition_sub_episodes:
                        self._episode_buffer.append(sub_ep)
                    # FIFO row-eviction; never drop the only remaining ep.
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
            # Cross-iter buffer or current-iter lists.
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

            # Snapshot for the trust-region KL-to-prev penalty.
            prev_policy: "GaussianPolicy | None" = None
            if cfg.prev_iter_kl_coef > 0.0:
                prev_policy = copy.deepcopy(self.policy)
                prev_policy.eval()
                for p in prev_policy.parameters():
                    p.requires_grad_(False)

            loss = self._distill_epoch(obs_flat, act_flat, w_flat, prev_policy=prev_policy)

            # Diagnostic KL, capped at 4096 obs.
            if prev_policy is not None and len(obs_flat) > 0:
                diag_n = min(len(obs_flat), 4096)
                diag_idx = np.random.choice(len(obs_flat), size=diag_n, replace=False)
                prev_iter_kl_diag = self.policy.kl_to_np(
                    obs_flat[diag_idx], prev_policy,
                )
            else:
                prev_iter_kl_diag = float("nan")

            # Stale Adam momentum fights prev_iter_kl_coef.
            if cfg.reset_optim_per_iter:
                self.policy.reset_optimizer()

            # ========== Logging ==========
            mppi_cost = float(np.mean(condition_costs))
            history.iteration_costs.append(mppi_cost)
            history.distill_losses.append(loss)
            history.iteration_prev_iter_kl.append(prev_iter_kl_diag)

            # Policy eval drives best.pt; NaN never overwrites.
            eval_every = max(int(getattr(cfg, "eval_every", 1)), 1)
            is_last = (iteration == num_iterations - 1)
            do_eval = is_last or ((iteration + 1) % eval_every == 0)

            eval_cost = float("nan")
            if do_eval:
                was_training = self.policy.training
                self.policy.eval()
                eval_stats = evaluate_policy(
                    self.policy, self.env,
                    n_episodes=cfg.n_eval_eps,
                    episode_len=cfg.eval_ep_len,
                    seed=iteration,
                )
                if was_training:
                    self.policy.train()
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
                    prev_iter_kl=prev_iter_kl_diag,
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
                f"[GPS iter {iteration:3d}]  "
                f"mppi_cost={mppi_cost:8.2f}  "
                f"distill_loss={loss:.4f}"
            )
            if cfg.distill_buffer_cap > 0:
                n_eps = len(self._episode_buffer)
                n_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                base_line += f"  buf={n_eps}eps/{n_rows}rows"
            if cfg.prev_iter_kl_coef > 0.0:
                base_line += f"  prev_kl={prev_iter_kl_diag:.4f}"
            if do_eval:
                base_line += f"  eval_cost={eval_cost:8.2f}"
            tqdm.write(base_line)

        outer_bar.close()
        return history
