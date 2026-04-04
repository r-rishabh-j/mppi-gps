import numpy as np
import optuna

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

# fixed
EVAL_STEPS = 500
N_SEEDS = 3


def objective(trial: optuna.Trial) -> float:
    # MPPI hyperparameters
    K = trial.suggest_int("K", 10, 100, log=True)
    H = trial.suggest_int("H", 4, 100, log=True)
    noise_sigma = trial.suggest_float("noise_sigma", 0.05, 0.3, log=True)
    lam = trial.suggest_float("lam", 0.05, 10.0, log=True)

   

    cfg = MPPIConfig(
        K=K,
        H=H,
        lam=lam,
        noise_sigma=noise_sigma,
        adaptive_lam=False,
        n_eff_threshold=max(16.0, K * 0.1),
    )

    total_cost = 0.0

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        env = Acrobot()

        # apply tuned cost weights
        # env._w_ctrl = w_ctrl
        # env._w_height = w_height
        # env._w_terminal = w_terminal

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

        # objective: how close is the tip to upright at the end?
        # also penalise residual velocity for stabilisation
        tip_z = env.data.site("tip").xpos[2]
        final_vel = np.sum(env.data.qvel ** 2)
        episode_cost = (4.0 - tip_z) ** 2 + 0.1 * final_vel

        trial.report(total_cost / (seed + 1), step=seed)
        if trial.should_prune():
            env.close()
            raise optuna.TrialPruned()

        env.close()
        total_cost += episode_cost

    return total_cost / N_SEEDS


def main():
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=50)

    print("\n=== Best trial ===")
    print("Best params:", study.best_params)
    print("Best cost:", study.best_value)


if __name__ == "__main__":
    main()
