import json
from pathlib import Path

import numpy as np
import optuna
from typing import NamedTuple
import functools

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

BEST_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "acrobot_best.json"


class FixedConfig(NamedTuple):
    n_startup_trials = 5
    EVAL_STEPS = 1000
    N_SEEDS = 10
    K = 256
    H = 256

def objective(trial: optuna.Trial, config: FixedConfig) -> float:
    # MPPI hyperparameters
    # K = trial.suggest_int("K", 2, 1000, log=True)
    # H = trial.suggest_int("H", 50, 500, log=True)
    noise_sigma = trial.suggest_float("noise_sigma", 0.01, 0.3, log=True)
    lam = trial.suggest_float("lam", 0.001, 1.0, log=True)

    cfg = MPPIConfig(
        K=config.K,
        H=config.H,
        lam=lam,
        noise_sigma=noise_sigma,
        adaptive_lam=False,
    )

    total_cost = 0.0
    env = Acrobot()
    controller = MPPI(env, cfg)
    for seed in range(config.N_SEEDS):
        env.reset()
        controller.reset()

        state = env.get_state()

        # track final performance (tip height + velocity)
        for t in range(config.EVAL_STEPS):
            action, info = controller.plan_step(state)
            obs, cost, done, _ = env.step(action)
            state = env.get_state()

            if done:
                break

            total_cost += cost

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
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3, n_startup_trials=config.n_startup_trials),
    )
    config_objective = functools.partial(objective, config=config)
    study.optimize(config_objective, n_trials=50, show_progress_bar=True)

    print("\n=== Best trial ===")
    print("Best params:", study.best_params)
    print("Best cost:", study.best_value)

    # save best params so run/visualisation scripts can load them
    best = study.best_params
    best["K"] = config.K
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"Saved best params to {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()
