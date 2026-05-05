"""MPPI-based Guided Policy Search (policy-prior-only variant).

Distills an MPPI controller into a reactive neural network policy via
behaviour cloning on MPPI rollouts collected under a policy-augmented cost.

The high-level loop is:
  1. For each initial condition, run MPPI with a policy-augmented cost
     that biases trajectory samples toward actions the policy can represent.
  2. Collect the executed (obs, action) pairs from each condition.
  3. Distill those pairs into the policy via supervised learning on the
     MPPI teacher's demonstrations (NLL or MSE).
  4. Optionally warm-start MPPI's nominal trajectory from the updated policy.
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


# ---------------------------------------------------------------------------
# Policy prior for MPPI (Eq. 5 from the proposal)
# ---------------------------------------------------------------------------

def make_policy_prior(policy: GaussianPolicy, env: BaseEnv, alpha: float):
    """Build a callable ``(states, actions) -> (K,)`` for MPPI's ``prior`` arg.

    ``MPPI.plan_step`` folds the prior return directly into the trajectory
    **cost** (not log-weights)::

        S_k     = S_env + is_corr + prior(states, actions)
        log w_k = -(S_k - rho) / lambda

    A *negative* prior return reduces S_k and increases the weight, so we
    return ``-alpha * sum_t log pi(u_t | obs_t)`` — biasing MPPI toward
    trajectories that are likely under the current policy. The effective
    contribution to ``log w_k`` is ``+(alpha / lambda) * sum_t log pi``.

    Args:
        policy: The current policy network (used in eval / no-grad mode).
        env:    Environment instance — needed for state_to_obs conversion.
        alpha:  Base weight on the policy augmentation term (from GPSConfig).
    """
    def prior_fn(states, actions, sensordata=None) -> np.ndarray:
        # states:     (K, H, nstate) — full physics states from batch_rollout
        # actions:    (K, H, act_dim) — clipped perturbed action sequences
        # sensordata: (K, H, nsensor) — sensor outputs paired with `states`
        #             (passed by mppi.plan_step; envs that don't need it ignore)
        states = np.asarray(states)
        actions = np.asarray(actions)
        K, H, _ = states.shape

        # Convert full physics states to policy-sized observations.
        obs = np.asarray(env.state_to_obs(states, sensordata))  # (K, H, obs_dim)

        # Evaluate log pi(u | obs) over the (K, H) grid in one batched call.
        obs_flat = obs.reshape(K * H, -1)
        act_flat = actions.reshape(K * H, -1)
        lp = policy.log_prob_np(obs_flat, act_flat)  # (K*H,)
        lp = lp.reshape(K, H).sum(axis=1)            # (K,) — sum over time

        # Negative-log-likelihood as a cost contribution.
        return -alpha * lp

    return prior_fn


# ---------------------------------------------------------------------------
# Main GPS class
# ---------------------------------------------------------------------------

@dataclass
class GPSHistory:
    """Training metrics collected during GPS, one entry per iteration."""
    # Mean C-step MPPI rollout cost (teacher-under-policy-prior) — diagnostic only.
    iteration_costs: list[float] = field(default_factory=list)
    # Greedy-policy eval cost; NaN on iterations where evaluation was skipped.
    # This is what `best.pt` is selected on.
    iteration_eval_costs: list[float] = field(default_factory=list)
    distill_losses: list[float] = field(default_factory=list)    # last distillation loss
    # Stabiliser diagnostics — NaN when the corresponding feature is off.
    iteration_ema_drift: list[float] = field(default_factory=list)    # ||θ − θ_ema||₂ after S-step
    iteration_prev_iter_kl: list[float] = field(default_factory=list) # E_obs[KL(π_new || π_prev)]
    best_iter: int = -1
    best_cost: float = float("inf")


class MPPIGPS:
    """MPPI-GPS: distills an MPPI controller into a neural network policy.

    The training loop iterates between:
      (C-step) Running MPPI with a policy-augmented cost on each initial
               condition to collect expert demonstrations.
      (S-step) Distilling those demonstrations into the policy via weighted
               maximum-likelihood (supervised learning).

    Pure policy-prior-only formulation: MPPI is biased by ``-alpha * log pi``
    in the trajectory cost, then BC follows. No BADMM/KL constraint.
    """

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

        # The global policy network that we are training.
        # obs_dim and act_dim come from the environment.
        self.policy = GaussianPolicy(
            env.obs_dim,
            env.action_dim,
            policy_cfg,
            device=device,
            action_bounds=env.action_bounds,
        )

        # A single MPPI controller instance, reused across conditions
        # (we reset its nominal U between conditions).
        self.mppi = MPPI(env, mppi_cfg)

        # Cross-iteration episode replay buffer. Each entry is one whole
        # sub-episode (dict with 'obs', 'actions', 'weights'), physically
        # contiguous (split at every `done` boundary during C-step). Empty
        # and unused when `gps_cfg.distill_buffer_cap == 0`.
        self._episode_buffer: list[dict] = []

        # ---- Stabilisers ----
        # EMA of the trainable policy params — tracked by the policy itself so
        # `train_weighted` / `mse_step` update the shadow after each Adam step
        # without the trainer having to hook in per-batch.
        if gps_cfg.ema_decay > 0.0:
            self.policy.attach_ema(gps_cfg.ema_decay)

        # `prev_iter_kl_coef` is applied inside the S-step via a deep-copied
        # snapshot taken at the start of each iteration's distillation loop;
        # no persistent state is needed here.

    # ----- helpers ---------------------------------------------------------

    def _sample_initial_conditions(self, n: int) -> list[np.ndarray]:
        """Sample n initial conditions by resetting the environment.

        Each reset produces a random initial state (the env's reset()
        randomises qpos/qvel).  We capture the full physics state via
        env.get_state() so we can restore it exactly later.

        Returns:
            List of n state vectors, each of shape (nstate,).
        """
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
        """Run multiple mini-batch gradient steps to distill MPPI into the policy.

        This is the S-step (supervised step) of GPS.  We train the policy
        to maximise the weighted log-likelihood of the MPPI teacher's actions:

            L = - Σ_i  w_i  log π_θ(a_i | o_i)
                + coef * KL(π_θ || π_prev)     (optional, NLL mode only)

        We do `distill_epochs` full passes over the data, shuffling each time,
        with mini-batches of size `distill_batch_size`.

        Args:
            obs:         (N, obs_dim)  — observations from all conditions.
            actions:     (N, act_dim)  — MPPI-executed actions.
            weights:     (N,)          — importance weights (uniform for now).
            prev_policy: Optional deep-copy of the policy at the start of this
                         S-step. When given and `gps_cfg.prev_iter_kl_coef > 0`
                         and `distill_loss == "nll"`, each batch's loss gains a
                         KL-to-previous-iteration penalty (trust-region style).
        Returns:
            The loss value from the last mini-batch.
        """
        cfg = self.gps_cfg
        N = len(obs)
        indices = np.arange(N)
        last_loss = 0.0
        # KL-to-prev is only defined for the Gaussian NLL path — MSE trains
        # only the mean and has no distribution to constrain.
        use_prev_kl = (
            prev_policy is not None
            and cfg.prev_iter_kl_coef > 0.0
            and cfg.distill_loss != "mse"
        )
        for _ in range(cfg.distill_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, cfg.distill_batch_size):
                batch = indices[start : start + cfg.distill_batch_size]
                if cfg.distill_loss == "mse":
                    last_loss = self.policy.mse_step(obs[batch], actions[batch])
                else:
                    # train_weighted computes -Σ w log π / Σ w  and does one Adam step
                    last_loss = self.policy.train_weighted(
                        obs[batch], actions[batch], weights[batch],
                        prev_policy=prev_policy if use_prev_kl else None,
                        prev_kl_coef=cfg.prev_iter_kl_coef if use_prev_kl else 0.0,
                    )
        return last_loss

    # def _train_step_mse(self, obs: np.ndarray, actions: np.ndarray) -> float:
    #     """Plain MSE on the policy mean — used when distill_loss='mse' (BC-style).

    #     Reuse GaussianPolicy.mse_step so the running observation normalizer is
    #     updated exactly the same way as the standalone BC and DAgger paths.
    #     """
    #     return self.policy.mse_step(obs, actions)

    def _warm_start_mppi(self, initial_state: np.ndarray):
        """Warm-start MPPI's nominal action sequence from the current policy.

        After the policy has been updated (S-step), we roll it out from the
        initial state to produce a sequence of H actions.  This becomes
        MPPI's starting nominal trajectory U for the next C-step, so that
        MPPI doesn't have to search from scratch — it refines around what
        the policy already knows.

        Args:
            initial_state: Full physics state to reset the environment to.
        """
        self.env.reset(state=initial_state)
        actions = []
        done = False
        for _ in range(self.mppi_cfg.H):
            if done:
                # Env has terminated — pad the remaining
                # horizon with zeros so we don't feed post-terminal garbage
                # states into the policy.
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
        """Run the full MPPI-GPS training loop.

        Each iteration:
          1. (C-step) For each initial condition, run MPPI with a policy-
             augmented prior for a full episode, collecting demonstrations.
          2. (S-step) Aggregate all (obs, action) pairs and distill them
             into the policy via mini-batched weighted MLE / MSE.

        Args:
            num_iterations: Override for gps_cfg.num_iterations.
            run_dir: If set, save per-iter wrapped checkpoints as
                `{run_dir}/iter_{k:03d}.pt` and copy the best-by-policy-eval
                to `{run_dir}/best.pt`. "Best" is decided from the greedy
                policy's eval cost (see `gps_cfg.n_eval_eps` / `eval_ep_len`
                / `eval_every`), not from the C-step MPPI rollout cost.
        Returns:
            GPSHistory with per-iteration metrics.
        """
        num_iterations = num_iterations or self.gps_cfg.num_iterations
        cfg = self.gps_cfg
        history = GPSHistory()

        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

        # Sample a fixed set of initial conditions that persist across all
        # iterations.  This ensures the policy is trained to handle a diverse
        # but consistent set of starting states.
        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)

        outer_bar = tqdm(range(num_iterations), desc="GPS", unit="iter")
        for iteration in outer_bar:
            alpha = cfg.policy_augmented_alpha

            # Accumulators for distillation data (aggregated across conditions)
            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []

            condition_costs: list[float] = []

            # ========== C-STEP: Run MPPI on each initial condition ==========
            # Nested progress bar over total MPPI steps (num_conditions * episode_length).
            # `leave=False` so it clears after each iteration and the outer bar stays clean.
            c_step_total = cfg.num_conditions * cfg.episode_length
            c_bar = tqdm(
                total=c_step_total,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )
            for ic_idx, ic_state in enumerate(initial_conditions):
                # Build the policy prior closure for this iteration.
                # This captures the *current* policy weights.
                prior_fn = make_policy_prior(self.policy, self.env, alpha)

                # After the first iteration, seed MPPI's nominal trajectory
                # from the policy so it starts from a better guess.  On iter 0
                # (or if warm-starting is off) reset U to zeros so nominal
                # state doesn't leak across initial conditions.
                if iteration > 0 and cfg.warm_start_policy:
                    self._warm_start_mppi(ic_state)
                else:
                    self.mppi.reset()

                # Reset environment to this initial condition
                self.env.reset(state=ic_state)

                # --- Sub-episode accumulators (one sub-episode closes at each
                #     `done`; without auto_reset there's exactly one sub-episode
                #     per condition). Each closed sub-episode is a contiguous
                #     slice of (obs, action) — never spans a reset.
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

                    # Run one MPPI planning step with the policy-augmented prior.
                    # This samples K trajectories, evaluates cost + prior, and
                    # returns the weighted-mean action. Mutates self.mppi.U and
                    # _last_* as usual — this is the "executor" call.
                    action_exec, info = self.mppi.plan_step(state, prior=prior_fn)

                    # DAgger-style relabel: fresh MPPI query at the SAME state
                    # with NO policy prior, used only as the training label.
                    # `dry_run=True` makes the call side-effect-free so the
                    # nominal-U warm start isn't disturbed. The env step below
                    # still uses action_exec, so the state distribution is
                    # steered by the prior — it's only the label that's
                    # uncontaminated.
                    if cfg.dagger_relabel and alpha > 0.0:
                        action_label, _ = self.mppi.plan_step(
                            state, prior=None, dry_run=True,
                        )
                    else:
                        action_label = action_exec

                    cur_obs.append(obs.copy())
                    cur_actions.append(action_label.copy())  # training target

                    # Step the environment with the exec action — state
                    # distribution is prior-biased, labels are not.
                    _, cost, done, _ = self.env.step(action_exec)
                    episode_cost += cost
                    c_bar.update(1)

                    if done and t < cfg.episode_length - 1:
                        # Close the current sub-episode BEFORE resetting so its
                        # contents reflect a single contiguous rollout.
                        _flush_sub_episode()
                        if cfg.auto_reset:
                            # Terminating env (hopper, etc.): re-seed to a fresh random
                            # init and keep collecting until episode_length steps are
                            # taken. MPPI nominal U is also reset so it doesn't carry
                            # post-fall state. Without this, the C-step would keep
                            # stepping MuJoCo from a fallen state and the S-step would
                            # distill on mostly-post-fall (obs, action) pairs.
                            self.env.reset()
                            self.mppi.reset()
                            continue
                        # No auto_reset: cut the episode short here. Pad the progress
                        # bar so totals stay consistent.
                        c_bar.update(cfg.episode_length - (t + 1))
                        break

                # Flush the tail sub-episode (the loop's last chunk, whether
                # it ran to `episode_length` or ended on the break above).
                _flush_sub_episode()

                # Route this condition's sub-episodes either into the buffer
                # (DAgger-style cross-iteration replay) or into the per-iter
                # lists (current on-policy-per-iter behaviour).
                if cfg.distill_buffer_cap > 0:
                    for sub_ep in condition_sub_episodes:
                        self._episode_buffer.append(sub_ep)
                    # FIFO eviction by rows — pop whole episodes from the front
                    # until total rows ≤ cap. Mirror DAgger's guard
                    # (`len(buffer) > 1`) so we never drop the *only* episode
                    # even if it alone exceeds the cap; otherwise the S-step
                    # would run with an empty buffer.
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

            # ========== S-STEP: Distill into policy ==========
            # Pull training data from either the cross-iteration buffer
            # (DAgger-style replay) or the current iteration's per-condition
            # lists. `distill_buffer_cap == 0` → current per-iter behaviour.
            if cfg.distill_buffer_cap > 0:
                obs_flat = np.concatenate(
                    [ep["obs"] for ep in self._episode_buffer], axis=0)
                act_flat = np.concatenate(
                    [ep["actions"] for ep in self._episode_buffer], axis=0)
                w_flat = np.concatenate(
                    [ep["weights"] for ep in self._episode_buffer], axis=0)
            else:
                obs_flat = np.concatenate(all_obs, axis=0)     # (total_steps, obs_dim)
                act_flat = np.concatenate(all_actions, axis=0)  # (total_steps, act_dim)
                w_flat = np.concatenate(all_weights, axis=0)    # (total_steps,)

            # Snapshot the policy at the start of the S-step for the
            # trust-region KL-to-prev penalty. The deep-copy shares no tensor
            # storage with `self.policy`, so subsequent gradient steps don't
            # drift the reference. We keep it even for iter 0 (the KL starts
            # at 0 and grows with drift — harmless) so the code path is
            # unconditional; the loss just applies it when coef > 0.
            prev_policy: "GaussianPolicy | None" = None
            if cfg.prev_iter_kl_coef > 0.0 and cfg.distill_loss != "mse":
                prev_policy = copy.deepcopy(self.policy)
                prev_policy.eval()
                for p in prev_policy.parameters():
                    p.requires_grad_(False)

            # Multiple epochs of mini-batch gradient descent
            loss = self._distill_epoch(obs_flat, act_flat, w_flat, prev_policy=prev_policy)

            # Diagnostics: post-distill EMA drift and KL to the pre-S-step
            # policy. Both default to NaN when the corresponding stabiliser
            # is disabled so plots show a clean "not tracked" signal.
            #
            # IMPORTANT ordering: compute BEFORE any hard-sync / optimizer
            # reset so the diagnostic reflects how far θ actually moved this
            # S-step. Post-sync, drift would always be 0 (θ == shadow).
            ema_drift = self.policy.ema_l2_drift() if cfg.ema_decay > 0.0 else float("nan")
            if prev_policy is not None and len(obs_flat) > 0:
                # Use at most 4096 obs rows to keep the diagnostic cheap.
                diag_n = min(len(obs_flat), 4096)
                diag_idx = np.random.choice(len(obs_flat), size=diag_n, replace=False)
                prev_iter_kl_diag = self.policy.kl_to_np(
                    obs_flat[diag_idx], prev_policy,
                )
            else:
                prev_iter_kl_diag = float("nan")

            # ========== End-of-S-step stabilisers ==========
            # 1. Hard-sync: promote the smoothed shadow to the live weights
            #    so the NEXT iteration's MPPI prior + S-step both start from
            #    the stable policy, not the noisy training trajectory.
            # 2. Adam reset: wipe m/v moments. Required for correctness after
            #    a hard-sync (moments were tied to pre-sync θ) and useful
            #    even without it — each GPS iter is a new supervised task,
            #    and stale momentum fights the `prev_iter_kl_coef` TR penalty.
            # Order matters: sync first so the reset targets the post-sync θ.
            if cfg.ema_decay > 0.0 and cfg.ema_hard_sync:
                self.policy.ema_sync()
            if cfg.reset_optim_per_iter:
                self.policy.reset_optimizer()

            # ========== Logging ==========
            mppi_cost = float(np.mean(condition_costs))
            history.iteration_costs.append(mppi_cost)
            history.distill_losses.append(loss)
            history.iteration_ema_drift.append(ema_drift)
            history.iteration_prev_iter_kl.append(prev_iter_kl_diag)

            # ========== Policy evaluation (drives best-checkpoint selection) ==========
            # We evaluate the student policy on fresh env resets and use THAT cost
            # to pick best.pt. The C-step rollout cost above is the MPPI teacher
            # under a policy prior — not a faithful measure of the student.
            #
            # Both eval AND the per-iter checkpoint save live inside the EMA
            # swap window, so:
            #   - the reported eval_cost reflects the smoothed policy;
            #   - iter_NNN.pt on disk contains the smoothed state_dict (what
            #     was evaluated), which in turn becomes best.pt via copy.
            # When EMA is disabled, ema_swapped_in() is a no-op context, so
            # behaviour is bit-for-bit identical to the pre-feature code path.
            eval_every = max(int(getattr(cfg, "eval_every", 1)), 1)
            is_last = (iteration == num_iterations - 1)
            do_eval = is_last or ((iteration + 1) % eval_every == 0)

            eval_cost = float("nan")
            with self.policy.ema_swapped_in():
                if do_eval:
                    was_training = self.policy.training
                    self.policy.eval()
                    eval_stats = evaluate_policy(
                        self.policy, self.env,
                        n_episodes=cfg.n_eval_eps,
                        episode_len=cfg.eval_ep_len,
                        seed=iteration,  # vary conditions across iters but deterministic per-iter
                    )
                    if was_training:
                        self.policy.train()
                    eval_cost = float(eval_stats["mean_cost"])
                history.iteration_eval_costs.append(eval_cost)

                # ========== Checkpointing ==========
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
                    # Best is selected on eval cost only (NaN comparisons are false,
                    # so skipped-eval iters never overwrite best.pt).
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
            if cfg.ema_decay > 0.0:
                base_line += f"  ema_drift={ema_drift:.4f}"
            if cfg.prev_iter_kl_coef > 0.0 and cfg.distill_loss != "mse":
                base_line += f"  prev_kl={prev_iter_kl_diag:.4f}"
            if do_eval:
                base_line += f"  eval_cost={eval_cost:8.2f}"
            tqdm.write(base_line)

        outer_bar.close()
        return history
