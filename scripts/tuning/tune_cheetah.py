import optuna 
import numpy as np 

from src.envs.half_cheetah import HalfCheetah
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

# fixed (match run_mppi.py)
K = 128
H = 10
eval_steps = 300
n_seeds = 2

def objective(trial: optuna.Trial) -> float:
    noise_sigma = trial.suggest_float("noise_sigma", 0.1, 3.0, log = True)
    lam = trial.suggest_float("lam", 0.001, 0.1, log = True)

    cfg = MPPIConfig(
        K = K,
        H = H,
        lam = lam,
        noise_sigma = noise_sigma,
        adaptive_lam = False,
    )

    total_cost = 0.0 

    for seed in range(n_seeds):
        # the seeds are for the noise that's being used within mppi 
        np.random.seed(seed)
        env = HalfCheetah()
        controller = MPPI(env, cfg)
        env.reset()
        state = env.get_state()

        episode_cost = 0.0 
        for t in range(eval_steps):
            action, info = controller.plan_step(state)
            obs, cost, done, _ = env.step(action)
            state = env.get_state()
            episode_cost += cost 

            # report for pruning 
            if t % 50 == 0 and t > 0:
                trial.report(episode_cost / (t + 1), step = t)
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()
            
            if done:
                break 
            
        env.close()
        total_cost += episode_cost
    
    return total_cost / n_seeds

def main():
    study = optuna.create_study(
        direction = "minimize", 
        pruner = optuna.pruners.MedianPruner(n_warmup_steps = 3),
    )

    study.optimize(objective, n_trials = 50)

    print("best params", study.best_params)
    print("best cost", study.best_value)

if __name__ == "__main__":
    main()











