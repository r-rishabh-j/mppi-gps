"""MPPI-based Guided Policy Search.

Distills an MPPI controller into a reactive neural network policy via
constrained optimization with BADMM dual updates on the KL divergence.

The high-level loop (Algorithm 1 from the proposal) is:
  1. For each initial condition, run MPPI with a policy-augmented cost
     that biases trajectory samples toward actions the policy can represent.
  2. Collect the executed (obs, action) pairs from each condition.
  3. Distill those pairs into the policy via weighted maximum-likelihood
     (supervised learning on the MPPI teacher's demonstrations).
  4. Measure the KL divergence between the MPPI trajectory distribution
     and the policy, then adjust the BADMM dual variable nu accordingly.
  5. Optionally warm-start MPPI's nominal trajectory from the updated policy.
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
from src.utils.math import weighted_mean_cov


# ---------------------------------------------------------------------------
# Policy prior for MPPI (Eq. 5 from the proposal)
# ---------------------------------------------------------------------------

def make_policy_prior(policy: GaussianPolicy, env: BaseEnv, alpha: float, nu: float):
    """Build a callable ``(states, actions) -> (K,)`` for MPPI's ``prior`` arg.

    MPPI now folds the return into the trajectory cost directly
    (``S = costs + is_corr + prior``, ``log w = -(S-rho)/lambda``), so we
    return ``-alpha * nu * sum_t log pi(u|o)`` — a cost contribution that
    rewards high-likelihood actions. See ``mppi_gps.make_policy_prior`` for
    the full derivation.
    """
    def prior_fn(states, actions) -> np.ndarray:
        states = np.asarray(states)
        actions = np.asarray(actions)
        K, H, _ = states.shape

        obs = np.asarray(env.state_to_obs(states))  # (K, H, obs_dim)
        obs_flat = obs.reshape(K * H, -1)
        act_flat = actions.reshape(K * H, -1)
        lp = policy.log_prob_np(obs_flat, act_flat)  # (K*H,)
        lp = lp.reshape(K, H).sum(axis=1)            # (K,) — sum over time

        return -alpha * nu * lp

    return prior_fn


# ---------------------------------------------------------------------------
# KL estimators
# ---------------------------------------------------------------------------

def _kl_diagonal_gaussian(mu_p, cov_p, mu_q, log_sigma_q):
    """Closed-form KL(N(mu_p, cov_p) || N(mu_q, diag(sigma_q^2)))."""
    D = mu_p.shape[0]
    var_q = np.exp(2.0 * log_sigma_q)                # (D,)
    inv_var_q = 1.0 / (var_q + 1e-8)                 # avoid division by zero

    trace_term = np.sum(np.diag(cov_p) * inv_var_q)
    diff = mu_q - mu_p
    mahal = np.sum(diff ** 2 * inv_var_q)

    log_det_q = 2.0 * np.sum(log_sigma_q)
    sign, log_det_p = np.linalg.slogdet(cov_p + 1e-8 * np.eye(D))
    log_det_ratio = log_det_q - log_det_p

    kl = 0.5 * (trace_term + mahal - D + log_det_ratio)
    return max(kl, 0.0)


def compute_kl_moment_matched(
    episode_mppi_actions: list[np.ndarray],
    episode_mppi_weights: list[np.ndarray],
    episode_obs: list[np.ndarray],
    policy: GaussianPolicy,
) -> tuple[float, dict]:
    """Moment-matched KL divergence (Eq. 3 from the proposal)."""
    kl_sum = 0.0
    T = len(episode_obs)

    for t in range(T):
        actions_t = episode_mppi_actions[t]    # (K, act_dim)
        weights_t = episode_mppi_weights[t]    # (K,)
        obs_t = episode_obs[t]                 # (obs_dim,)

        mu_mppi, cov_mppi = weighted_mean_cov(actions_t, weights_t)

        device = getattr(policy, "_device", torch.device("cpu"))
        obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu_pi, log_sigma_pi = policy(obs_tensor)
        mu_pi = mu_pi.squeeze(0).cpu().numpy()
        log_sigma_pi = log_sigma_pi.squeeze(0).cpu().numpy()

        kl_sum += _kl_diagonal_gaussian(mu_mppi, cov_mppi, mu_pi, log_sigma_pi)

    kl_avg = kl_sum / max(T, 1)
    return kl_avg, {"kl_sum": kl_sum, "T": T}


def compute_kl_sample_based(
    episode_mppi_actions: list[np.ndarray],
    episode_mppi_weights: list[np.ndarray],
    episode_obs: list[np.ndarray],
    policy: GaussianPolicy,
) -> tuple[float, dict]:
    """Sample-based KL estimate (Eq. 4 from the proposal)."""
    kl_sum = 0.0
    T = len(episode_obs)

    for t in range(T):
        actions_t = episode_mppi_actions[t]    # (K, act_dim)
        weights_t = episode_mppi_weights[t]    # (K,)
        obs_t = episode_obs[t]                 # (obs_dim,)

        K = actions_t.shape[0]
        obs_batch = np.broadcast_to(obs_t, (K, obs_t.shape[-1]))
        log_pi = policy.log_prob_np(obs_batch, actions_t)  # (K,)

        log_w = np.log(weights_t + 1e-10)  # +eps to avoid log(0)
        kl_t = np.sum(weights_t * (log_w - log_pi))
        kl_sum += max(kl_t, 0.0)

    kl_avg = kl_sum / max(T, 1)
    return kl_avg, {"kl_sum": kl_sum, "T": T}


# ---------------------------------------------------------------------------
# Main GPS class
# ---------------------------------------------------------------------------

@dataclass
class GPSHistory:
    """Training metrics collected during GPS, one entry per iteration."""
    iteration_costs: list[float] = field(default_factory=list)
    iteration_eval_costs: list[float] = field(default_factory=list)
    iteration_kl: list[float] = field(default_factory=list)
    iteration_nu: list[float] = field(default_factory=list)
    distill_losses: list[float] = field(default_factory=list)
    best_iter: int = -1
    best_cost: float = float("inf")


class MPPIGPS:
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

        self.nu = gps_cfg.badmm_init_nu
        self.mppi = MPPI(env, mppi_cfg)
        self._episode_buffer: list[dict] = []

    # ----- helpers ---------------------------------------------------------

    def _sample_initial_conditions(self, n: int) -> list[np.ndarray]:
        conditions = []
        for _ in range(n):
            self.env.reset()
            conditions.append(self.env.get_state().copy())
        return conditions

    def _train_step_clipped(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
        old_policy,
        clip_ratio: float = 0.2,
        clip_eps: float = 0.1
    ) -> float:
        """Trust-region update for the S-Step (student policy)."""
        device = self.policy.device
        
        # Determine if we are doing MSE (Deterministic) or NLL (Gaussian)
        if self.gps_cfg.distill_loss == "mse" or not hasattr(self.policy, "log_prob"):
            # --- Deterministic Policy / MSE Path ---
            with torch.no_grad():
                o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                old_pred = old_policy.action(o_t).cpu().numpy()

            clipped_actions = np.clip(actions, old_pred - clip_eps, old_pred + clip_eps)
            return self.policy.mse_step(obs, clipped_actions)
            
        else:
            # --- Gaussian Policy / Weighted MLE Path ---
            o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            a_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
            w_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

            if self.policy.normalizer is not None:
                self.policy.normalizer.update(o_t)

            curr_log_prob = self.policy.log_prob(o_t, a_t)
            with torch.no_grad():
                old_log_prob = old_policy.log_prob(o_t, a_t)

            ratio = torch.exp(curr_log_prob - old_log_prob)
            
            # PPO-style clipping weighted by the MPPI importance weights
            surr1 = ratio * w_t
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * w_t
            
            # Loss matches train_weighted: negative sum normalized by total weight
            loss = -torch.min(surr1, surr2).sum() / w_t.sum().clamp(min=1e-8)

            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
            return float(loss.item())

    def _distill_epoch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Run multiple mini-batch gradient steps to distill MPPI into the policy."""
        cfg = self.gps_cfg
        N = len(obs)
        indices = np.arange(N)
        last_loss = 0.0
        
        # Snapshot the old policy for clipping
        old_policy = copy.deepcopy(self.policy)
        old_policy.eval()
        
        clip_ratio = getattr(cfg, "clip_ratio", 0.2)
        clip_eps = getattr(cfg, "clip_eps", 0.1)

        for _ in range(cfg.distill_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, cfg.distill_batch_size):
                batch = indices[start : start + cfg.distill_batch_size]
                
                # Replace the simple unconstrained step with the clipped step
                last_loss = self._train_step_clipped(
                    obs[batch], 
                    actions[batch], 
                    weights[batch],
                    old_policy=old_policy,
                    clip_ratio=clip_ratio,
                    clip_eps=clip_eps
                )
        return last_loss

    def _update_badmm(self, kl_value: float):
        """Adjust the BADMM dual variable nu based on the current KL."""
        target = self.gps_cfg.kl_target
        step = self.gps_cfg.badmm_step_size
        if kl_value > target:
            self.nu *= step      # tighten: increase the penalty
        elif kl_value < target:
            self.nu /= step      # relax: decrease the penalty
        self.nu = np.clip(self.nu, 1e-4, 1e4)

    def _warm_start_mppi(self, initial_state: np.ndarray):
        """Warm-start MPPI's nominal action sequence from the current policy."""
        self.env.reset(state=initial_state)
        actions = []
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
        """Run the full MPPI-GPS training loop."""
        num_iterations = num_iterations or self.gps_cfg.num_iterations
        cfg = self.gps_cfg
        history = GPSHistory()

        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

        initial_conditions = self._sample_initial_conditions(cfg.num_conditions)
        skip_kl = cfg.disable_kl_constraint

        outer_bar = tqdm(range(num_iterations), desc="GPS", unit="iter")
        for iteration in outer_bar:
            alpha = cfg.policy_augmented_alpha

            all_obs: list[np.ndarray] = []
            all_actions: list[np.ndarray] = []
            all_weights: list[np.ndarray] = []

            all_kl_actions: list[list[np.ndarray]] = []
            all_kl_weights: list[list[np.ndarray]] = []
            all_kl_obs: list[list[np.ndarray]] = []

            condition_costs: list[float] = []

            c_step_total = cfg.num_conditions * cfg.episode_length
            c_bar = tqdm(
                total=c_step_total,
                desc=f"  iter {iteration:3d} C-step",
                unit="step",
                leave=False,
            )
            for ic_idx, ic_state in enumerate(initial_conditions):
                prior_fn = make_policy_prior(
                    self.policy, self.env, alpha, self.nu,
                )

                if iteration > 0 and cfg.warm_start_policy:
                    self._warm_start_mppi(ic_state)
                else:
                    self.mppi.reset()

                self.env.reset(state=ic_state)

                condition_sub_episodes: list[dict] = []
                cur_obs: list[np.ndarray] = []
                cur_actions: list[np.ndarray] = []
                episode_kl_actions = []   
                episode_kl_weights = []   
                episode_obs_for_kl: list[np.ndarray] = []  
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

                    action, info = self.mppi.plan_step(state, prior=prior_fn)

                    cur_obs.append(obs.copy())
                    cur_actions.append(action.copy())  

                    if not skip_kl:
                        rollout_data = self.mppi.get_rollout_data()
                        first_step_actions = np.array(rollout_data['actions'][:, 0, :])
                        weights = np.array(rollout_data['weights'])
                        episode_kl_actions.append(first_step_actions)
                        episode_kl_weights.append(weights)
                        episode_obs_for_kl.append(obs.copy())

                    _, cost, done, _ = self.env.step(action)
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

                if not skip_kl:
                    all_kl_actions.append(episode_kl_actions)
                    all_kl_weights.append(episode_kl_weights)
                    all_kl_obs.append(episode_obs_for_kl)

                condition_costs.append(episode_cost)
                c_bar.set_postfix(
                    ic=f"{ic_idx + 1}/{cfg.num_conditions}",
                    last_cost=f"{episode_cost:.1f}",
                )
            c_bar.close()

            # ========== S-STEP: Distill into policy ==========
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

            # ========== KL computation (average across conditions) ==========
            if skip_kl:
                kl_mean = float("nan")
            else:
                kl_values = []
                kl_fn = (compute_kl_moment_matched
                         if cfg.kl_estimator == "moment_matched"
                         else compute_kl_sample_based)
                for ic_idx in range(cfg.num_conditions):
                    kl_val, _ = kl_fn(
                        all_kl_actions[ic_idx],
                        all_kl_weights[ic_idx],
                        all_kl_obs[ic_idx],
                        self.policy,
                    )
                    kl_values.append(kl_val)
                kl_mean = float(np.mean(kl_values))
                self._update_badmm(kl_mean)

            # ========== Logging ==========
            mppi_cost = float(np.mean(condition_costs))
            history.iteration_costs.append(mppi_cost)
            history.iteration_kl.append(kl_mean)
            history.iteration_nu.append(self.nu)
            history.distill_losses.append(loss)

            # ========== Policy evaluation ==========
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

            # ========== Checkpointing ==========
            if run_dir is not None:
                iter_path = run_dir / f"iter_{iteration:03d}.pt"
                save_checkpoint(
                    iter_path, self.policy,
                    iteration=iteration,
                    mppi_cost=mppi_cost,
                    eval_cost=eval_cost,
                    distill_loss=loss,
                    kl=kl_mean,
                    nu=self.nu,
                )
                if do_eval and eval_cost < history.best_cost:
                    history.best_cost = eval_cost
                    history.best_iter = iteration
                    copy_as(iter_path, run_dir / "best.pt")

            postfix = {
                "mppi_cost": f"{mppi_cost:.2f}",
                "loss": f"{loss:.3f}",
                "kl": f"{kl_mean:.3f}",
                "nu": f"{self.nu:.3f}",
                "best": history.best_iter if history.best_iter >= 0 else "-",
            }
            if do_eval:
                postfix["eval_cost"] = f"{eval_cost:.2f}"
            outer_bar.set_postfix(**postfix)

            base_line = (
                f"[GPS iter {iteration:3d}]  "
                f"mppi_cost={mppi_cost:8.2f}  "
                f"distill_loss={loss:.4f}  "
                f"kl={kl_mean:.4f}  "
                f"nu={self.nu:.4f}"
            )
            if cfg.distill_buffer_cap > 0:
                n_eps = len(self._episode_buffer)
                n_rows = sum(len(ep["obs"]) for ep in self._episode_buffer)
                base_line += f"  buf={n_eps}eps/{n_rows}rows"
            if do_eval:
                base_line += f"  eval_cost={eval_cost:8.2f}"
            tqdm.write(base_line)

        outer_bar.close()
        return history