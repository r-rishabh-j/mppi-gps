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
python -m scripts.run_gps --env acrobot --device auto
python -m scripts.run_gps --env hopper --gps-iters 30 --device auto

# Policy-prior-only GPS (no KL / no BADMM — MPPI biased by -α·log π, then plain BC)
python -m scripts.run_gps --env acrobot --disable-kl --distill-loss mse \
    --alpha 0.1 --nu 1.0 --device auto

# DAgger (MPPI-in-the-loop BC; policy trains on GPU, MPPI stays on CPU)
python -m scripts.run_dagger --env acrobot --dagger-iters 10 \
    --rollouts-per-iter 20 --episode-len 200 \
    --seed-from data/acrobot_bc.h5 --device auto

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

- **`src/mppi/mppi.py`** — Core MPPI controller (CPU/numpy). Samples K trajectory perturbations, computes importance-weighted updates to a nominal action sequence, and supports adaptive temperature (λ) to maintain effective sample size.
- **`src/envs/`** — Environment implementations behind an abstract `BaseEnv` interface (`base.py`). `mujoco_env.py` wraps MuJoCo with multi-threaded batch rollout (CPU). `gym_wrapper.py` adapts Gymnasium envs. Concrete tasks: `point_mass.py`, `half_cheetah.py`, `acrobot.py`, `hopper.py`. Each env provides `state_to_obs()` for converting full physics state to policy observations. Factory: `src.envs.make_env(name)`.
- **`src/policy/gaussian_policy.py`** — Diagonal Gaussian MLP policy for behavior cloning and GPS distillation. Accepts `device=` (cpu/mps/cuda); `act_np()` / `log_prob_np()` bridge to numpy callers.
- **`src/gps/mppi_gps.py`** — Core GPS training loop (`MPPIGPS` class). Distills MPPI into the policy via BADMM-constrained optimization with policy-augmented MPPI cost and moment-matched or sample-based KL estimators. Set `GPSConfig.disable_kl_constraint=True` (or `--disable-kl`) for the policy-prior-only variant: MPPI biased by `α·nu·log π(u|s)` with `nu` held constant, followed by plain BC (NLL or MSE via `distill_loss`).
- **`src/gps/dagger.py`** — `DAggerTrainer`: roll out the current policy (with β-mixed MPPI execution), relabel every visited state via `MPPI.plan_step`, aggregate into a capped buffer, finetune the policy. MPPI stays on CPU; policy trains on the selected device.
- **`src/gps/ilqr.py`** — GPS via iLQR (skeleton/in-progress).
- **`src/utils/config.py`** — Dataclass configs (`MPPIConfig`, `PolicyConfig`, `GPSConfig`, `DAggerConfig`).
- **`src/utils/device.py`** — `pick_device("auto"|"cpu"|"mps"|"cuda")` resolves `auto` → cuda → mps → cpu.
- **`src/utils/math.py`** — Numerically stable log-sum-exp, weight computation, KL helpers (numpy).
- **`src/utils/evaluation.py`** — Shared `evaluate_policy` and `evaluate_mppi` used by GPS and BC scripts.
- **`configs/`** — JSON hyperparameter configs (e.g., `acrobot_best.json`).
- **`assets/`** — MuJoCo XML model files.

## Keeping docs in sync

`commands.md` is the authoritative, user-facing cheat sheet for runnable commands
(setup, MPPI, BC, DAgger, GPS, checkpoint eval, tuning, ablations). When you add
or change a CLI flag, rename a script, or introduce a new training variant,
update `commands.md` in the same change — the brief command list in this file
stays as a high-level overview; detailed flag docs live in `commands.md`.

## Key Patterns

- States are `(K, H, state_dim)` arrays, actions `(K, H, action_dim)`, costs `(K,)` — heavy NumPy broadcasting throughout.
- MPPI batch rollouts use MuJoCo's native `rollout` API with a thread pool (CPU only).
- Policy training via PyTorch runs on the device selected with `--device auto|cpu|mps|cuda` (GPS, DAgger, and BC scripts all take `--device`). MPPI always runs on CPU; only the policy crosses onto an accelerator.
- Python >=3.11 required. Dependency management via `uv` with `uv.lock` for reproducibility.