import optuna
import numpy as np
from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

# fixed
K = 2048
H = 100
EVAL_STEPS = 500
N_SEEDS = 2


def objective(trial: optuna.Trial) -> float:
    noise_sigma = trial.suggest_float("noise_sigma", 0.1, 2.0, log=True)
    lam = trial.suggest_float("lam", 0.01, 10.0, log=True)
    P_scale = trial.suggest_float("P_scale", 100.0, 10000.0, log=True)

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
        env._P_scale = P_scale
        controller = MPPI(env, cfg)
        env.reset()
        state = env.get_state()

        episode_cost = 0.0
        for t in range(EVAL_STEPS):
            action, info = controller.plan_step(state)
            obs, cost, done, _ = env.step(action)
            state = env.get_state()
            episode_cost += cost

            if t % 100 == 0 and t > 0:
                trial.report(episode_cost / (t + 1), step=t)
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()

            if done:
                break

        env.close()
        total_cost += episode_cost

    return total_cost / N_SEEDS


def main():
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=50)

    print("Best params:", study.best_params)
    print("Best cost:", study.best_value)


if __name__ == "__main__":
    main()
