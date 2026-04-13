"""Optuna-based hyperparameter tuning for MPPI on the Hopper environment.

Searches over noise_sigma and lambda (temperature) while keeping K and H
fixed.  Uses median pruning to early-stop unpromising trials.

The objective is the mean total cost across N_SEEDS episodes.  If the hopper
falls (done=True), we penalise the remaining steps to discourage configs
that lead to instability.

Usage:
    python scripts/tuning/tune_hopper.py

Best parameters are saved to configs/hopper_best.json.
"""

import json
from pathlib import Path

import numpy as np
import optuna
from typing import NamedTuple
import functools

from src.envs.hopper import Hopper
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

BEST_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "hopper_best.json"


class FixedConfig(NamedTuple):
    """Non-tuned parameters that stay fixed across all trials."""
    n_startup_trials = 5    # random trials before GP kicks in
    EVAL_STEPS = 500        # steps per evaluation episode
    N_SEEDS = 5             # episodes per trial (averaged for objective)
    K = 512                 # MPPI sample count (fixed — not tuned)
    H = 64                  # planning horizon (fixed — not tuned)


def objective(trial: optuna.Trial, config: FixedConfig) -> float:
    """Optuna objective: mean cost across N_SEEDS episodes.

    Tuned parameters:
      - noise_sigma: exploration noise std in [0.05, 1.0] (log scale)
      - lam: MPPI temperature in [0.01, 10.0] (log scale)
    """
    noise_sigma = trial.suggest_float("noise_sigma", 0.05, 1.0, log=True)
    lam = trial.suggest_float("lam", 0.01, 10.0, log=True)

    cfg = MPPIConfig(
        K=config.K,
        H=config.H,
        lam=lam,
        noise_sigma=noise_sigma,
        adaptive_lam=True,           # let MPPI auto-adjust lambda
        n_eff_threshold=128,
    )

    total_cost = 0.0
    env = Hopper()
    controller = MPPI(env, cfg)
    for seed in range(config.N_SEEDS):
        env.reset()
        controller.reset()
        state = env.get_state()

        for t in range(config.EVAL_STEPS):
            action, info = controller.plan_step(state)
            obs, cost, done, _ = env.step(action)
            state = env.get_state()

            if done:
                # Penalise falling: charge the current cost for all remaining
                # steps so that unstable configs get a high total cost.
                total_cost += cost * (config.EVAL_STEPS - t)
                break

            total_cost += cost

        # Report intermediate result so the pruner can early-stop bad trials
        trial.report(total_cost / (seed + 1), step=seed)
        if trial.should_prune():
            env.close()
            raise optuna.TrialPruned()

    env.close()

    return total_cost / config.N_SEEDS


def main():
    config = FixedConfig()
    study = optuna.create_study(
        direction="minimize",
        # GP sampler uses a Gaussian process surrogate for Bayesian optimisation
        sampler=optuna.samplers.GPSampler(n_startup_trials=config.n_startup_trials),
        # Median pruner stops trials that are worse than the median at the
        # same intermediate step
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=config.n_startup_trials),
    )
    config_objective = functools.partial(objective, config=config)
    study.optimize(config_objective, n_trials=50, show_progress_bar=True)

    print("\n=== Best trial ===")
    print("Best params:", study.best_params)
    print("Best cost:", study.best_value)

    # Save best params (merge tuned values with fixed K, H, etc.)
    best = study.best_params
    best["K"] = config.K
    best["H"] = config.H
    best["adaptive_lam"] = True
    best["n_eff_threshold"] = 128
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"Saved best params to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
