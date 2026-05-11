"""Optuna-based hyperparameter tuning for MPPI on the PointMass environment.

Searches over noise_sigma and lambda (temperature) while keeping K and H
fixed. Uses median pruning to early-stop unpromising trials. The
objective is the mean total cost across N_SEEDS episodes — each episode
samples a fresh random goal (see `PointMass._sample_goal`), so the tuned
hyperparameters are robust across the whole goal sampling box rather
than overfit to a single (0, 0) target.

Usage:
    python scripts/tuning/tune_point_mass.py

Best parameters are saved to configs/point_mass_best.json.
"""

import json
from pathlib import Path

import optuna
from typing import NamedTuple
import functools

from src.envs.point_mass import PointMass
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

BEST_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "point_mass_best.json"


class FixedConfig(NamedTuple):
    """Non-tuned parameters that stay fixed across all trials."""
    n_startup_trials: int = 5    # random trials before GP kicks in
    EVAL_STEPS: int = 300        # steps per evaluation episode (env dt=0.02 → 6s)
    N_SEEDS: int = 10            # episodes per trial (random goal each), averaged
    K: int = 256                 # MPPI sample count (fixed — not tuned)
    H: int = 64                  # planning horizon (fixed — not tuned)


def objective(trial: optuna.Trial, config: FixedConfig) -> float:
    """Optuna objective: mean cost across N_SEEDS episodes.

    Tuned parameters:
      - noise_sigma: exploration noise std in [0.05, 1.5] (log scale)
      - lam:         MPPI temperature in [0.001, 1.0] (log scale)
    """
    noise_sigma = trial.suggest_float("noise_sigma", 0.05, 1.5, log=True)
    lam = trial.suggest_float("lam", 0.001, 1.0, log=True)

    cfg = MPPIConfig(
        K=config.K,
        H=config.H,
        lam=lam,
        noise_sigma=noise_sigma,
        adaptive_lam=False,
    )

    total_cost = 0.0
    env = PointMass()
    controller = MPPI(env, cfg)
    for seed in range(config.N_SEEDS):
        env.reset()           # new random goal each episode
        controller.reset()
        state = env.get_state()

        for t in range(config.EVAL_STEPS):
            action, info = controller.plan_step(state)
            obs, cost, done, _ = env.step(action)
            state = env.get_state()
            total_cost += cost
            if done:
                break

        # Report intermediate result so the pruner can early-stop bad trials.
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
        sampler=optuna.samplers.GPSampler(n_startup_trials=config.n_startup_trials),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=3, n_startup_trials=config.n_startup_trials
        ),
    )
    config_objective = functools.partial(objective, config=config)
    study.optimize(config_objective, n_trials=50, show_progress_bar=True)

    print("\n=== Best trial ===")
    print("Best params:", study.best_params)
    print("Best cost:", study.best_value)

    # Merge tuned values with the fixed K/H so a downstream load
    # (`MPPIConfig.load("point_mass")`) gets a complete config.
    best = study.best_params
    best["K"] = config.K
    best["H"] = config.H
    best["adaptive_lam"] = False
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"Saved best params to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
