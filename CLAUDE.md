# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase implementing **Model Predictive Path Integral (MPPI)** control with **Guided Policy Search (GPS)** for policy distillation. Uses MuJoCo for physics simulation across point-mass, half-cheetah, acrobot, and hopper environments.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run MPPI control on different environments
python -m scripts.run_point_mass
python -m scripts.run_acrobot
python -m scripts.run_cheetah
python -m scripts.run_mppi

# Collect behavioral cloning demonstrations from MPPI
python -m scripts.collect_bc_demos

# Train and test supervised learning policy
python -m scripts.test_sl

# GPS training (distills MPPI into reactive policy)
python -m scripts.run_gps --env acrobot
python -m scripts.run_gps --env hopper --gps-iters 30

# Hyperparameter tuning (Optuna)
python -m scripts.tuning.tune_acrobot
python -m scripts.tuning.tune_cheetah
python -m scripts.tuning.tune_hopper

# SB3 baselines
python -m scripts.run_sb3_baseline --env Hopper-v5 --algo SAC

# Ablations and plotting
python -m scripts.run_ablations --env acrobot
python -m scripts.visualisation.plot_results --env acrobot
```

No unit test framework is configured. Validation is done via `scripts/test_sl.py` which compares policy performance against the MPPI baseline.

## Architecture

- **`src/mppi/mppi.py`** — Core MPPI controller. Samples K trajectory perturbations, computes importance-weighted updates to a nominal action sequence, and supports adaptive temperature (λ) to maintain effective sample size.
- **`src/envs/`** — Environment implementations behind an abstract `BaseEnv` interface (`base.py`). `mujoco_env.py` wraps MuJoCo with multi-threaded batch rollout. `gym_wrapper.py` adapts Gymnasium envs. Concrete tasks: `point_mass.py`, `half_cheetah.py`, `acrobot.py`, `hopper.py`. Each env provides `state_to_obs()` for converting full physics state to policy observations.
- **`src/policy/gaussian_policy.py`** — Diagonal Gaussian MLP policy for behavior cloning and GPS distillation.
- **`src/gps/mppi_gps.py`** — Core GPS training loop (`MPPIGPS` class). Distills MPPI into the policy via BADMM-constrained optimization with policy-augmented MPPI cost and moment-matched or sample-based KL estimators.
- **`src/gps/ilqr.py`** — GPS via iLQR (skeleton/in-progress).
- **`src/utils/config.py`** — Dataclass configs (`MPPIConfig`, `PolicyConfig`, `GPSConfig`).
- **`src/utils/math.py`** — Numerically stable log-sum-exp, weight computation, KL helpers.
- **`src/utils/evaluation.py`** — Shared `evaluate_policy` and `evaluate_mppi` used by GPS and BC scripts.
- **`configs/`** — JSON hyperparameter configs (e.g., `acrobot_best.json`).
- **`assets/`** — MuJoCo XML model files.

## Key Patterns

- States are `(K, H, state_dim)` arrays, actions `(K, H, action_dim)`, costs `(K,)` — heavy NumPy broadcasting throughout.
- Batch rollouts use MuJoCo's native `rollout` API with thread pool for parallel trajectory simulation.
- `jaxtyping` annotations document array shapes (JAX is used only for type hints, not computation).
- Python >=3.11 required. Dependency management via `uv` with `uv.lock` for reproducibility.