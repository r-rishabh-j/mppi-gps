import numpy as np
import optuna

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

# fixed
EVAL_STEPS = 500
N_SEEDS = 10
H = 128

def objective(trial: optuna.Trial) -> float:
    # MPPI hyperparameters
    K = trial.suggest_int("K", 2, 500, log=True)
    # H = trial.suggest_int("H", 10, 100, log=True)
    noise_sigma = trial.suggest_float("noise_sigma", 0.001, 0.3, log=True)
    lam = trial.suggest_float("lam", 0.0001, 5.0, log=True)

    cfg = MPPIConfig(
        K=K,
        H=H,
        lam=lam,
        noise_sigma=noise_sigma,
        adaptive_lam=False,
    )

    total_cost = 0.0

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        env = Acrobot()

        controller = MPPI(env, cfg)
        env.reset()

        state = env.get_state()

        # track final performance (tip height + velocity)
        for t in range(EVAL_STEPS):
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
        

    return total_cost / N_SEEDS


def main():
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.GPSampler(), 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=50)

    print("\n=== Best trial ===")
    print("Best params:", study.best_params)
    print("Best cost:", study.best_value)


if __name__ == "__main__":
    main()
