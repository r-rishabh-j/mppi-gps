# MPPI-GPS Implementation Report

## Overview

This document describes the implementation of the MPPI-based Guided Policy Search (GPS) system that distills an online MPPI trajectory optimiser into a cheap, reactive neural network policy. The core idea is: MPPI produces excellent trajectories at high computational cost (K rollouts per step); GPS trains a feedforward policy to mimic MPPI's behaviour, enabling real-time deployment without the simulator-in-the-loop.

---

## What Was Implemented

### Phase 1: Core GPS Loop

#### 1. `state_to_obs` and `obs_dim` on all environments

**Files modified:** `src/envs/base.py`, `src/envs/mujoco_env.py`, `src/envs/acrobot.py`, `src/envs/half_cheetah.py`, `src/envs/point_mass.py`, `src/envs/gym_wrapper.py`

MPPI works with **full physics states** (from `mj_getState`: `[time, qpos, qvel, act, ...]`) but the policy network works with **observations** (e.g. `[qpos, qvel]` for Acrobot, or `[qpos[1:], qvel]` for HalfCheetah where root x is excluded).

The new `state_to_obs(states)` method converts batched full-state arrays `(..., nstate)` to policy-sized observations `(..., obs_dim)`, using the existing `state_qpos` / `state_qvel` helpers that correctly handle the time offset at index 0.

| Environment | obs_dim | Observation contents |
|---|---|---|
| Acrobot | 4 | qpos (2) + qvel (2) |
| HalfCheetah | 17 | qpos[1:] (8) + qvel (9) — skip root x |
| PointMass | 4 | qpos (2) + qvel (2) |
| Hopper | 11 | qpos[1:] (5) + clip(qvel, -10, 10) (6) |

#### 2. `src/gps/mppi_gps.py` — Core GPS training class

This is the central new file. It contains:

**`make_policy_prior(policy, env, alpha, nu)`** — Builds the callable that plugs into MPPI's `prior` argument. Returns `+alpha * nu * sum_t log π(u_t | obs_t)` for each of the K trajectory samples. This biases MPPI toward actions the policy can represent (Eq. 5 from the proposal). The sign is positive because MPPI's weight formula is `log_w = -cost/λ + log_prior`, so a positive prior *reduces* the effective cost.

**`_kl_diagonal_gaussian(mu_p, cov_p, mu_q, log_sigma_q)`** — Closed-form KL divergence from a full-covariance Gaussian (moment-matched MPPI distribution) to a diagonal Gaussian (policy output). Uses the standard formula:

```
KL(p||q) = 0.5 * [tr(Σ_q⁻¹ Σ_p) + (μ_q - μ_p)ᵀ Σ_q⁻¹ (μ_q - μ_p) - D + ln(det Σ_q / det Σ_p)]
```

Since Σ_q is diagonal, the inverse and log-determinant reduce to element-wise operations.

**`compute_kl_moment_matched()`** — Eq. 3 estimator. At each timestep t:
1. Fits `N(μ_t, Σ_t)` to the K weighted MPPI first-step actions via `weighted_mean_cov`.
2. Queries the policy for `π_θ(·|obs_t) = N(μ_π, σ_π²I)`.
3. Computes closed-form KL and averages over timesteps.

**`compute_kl_sample_based()`** — Eq. 4 estimator. Uses the identity `KL ≈ Σ_k w_k (log w_k - log π(u_k|x_k))` directly on the weighted particles, without fitting a Gaussian. Higher variance but captures multi-modal distributions.

**`MPPIGPS` class** — The main training loop, alternating between:
- **C-step** (controller step): For each initial condition, run MPPI with the policy-augmented prior for a full episode. Collect executed (obs, action) pairs and per-timestep MPPI sample distributions.
- **S-step** (supervised step): Aggregate all (obs, action) pairs across conditions, then run multiple epochs of mini-batched weighted MLE to distill into the policy.
- **BADMM update**: If KL > target, multiply nu by step_size (tighten); if KL < target, divide (relax). Clamped to [1e-4, 1e4].
- **Warm-start**: After the first iteration, roll out the policy from each initial condition to seed MPPI's nominal U, so MPPI refines rather than searches from scratch.

#### 3. `GPSConfig` extensions

**File modified:** `src/utils/config.py`

New fields added to `GPSConfig`:
- `episode_length: int = 500` — steps per episode during GPS training
- `kl_target: float = 1.0` — target KL for BADMM dual update
- `distill_batch_size: int = 256` — mini-batch size for policy gradient steps
- `distill_epochs: int = 5` — number of gradient passes per GPS iteration
- `warm_start_policy: bool = True` — whether to seed MPPI from the policy

#### 4. `src/utils/evaluation.py` — Shared evaluation helpers

**New file.** Extracted `evaluate_policy()` and `evaluate_mppi()` from `scripts/test_sl.py` into a shared module. Both functions use the same seed schedule `(seed + ep)` so that episode i starts from identical initial conditions, making per-episode cost comparisons apples-to-apples.

`scripts/test_sl.py` was updated to import from this shared module instead of defining its own copies.

#### 5. `scripts/run_gps.py` — Main GPS training script

**New file.** Entry point that:
1. Creates the environment and loads tuned MPPI hyperparameters from `configs/<env>_best.json`.
2. Instantiates `MPPIGPS` and runs `train()`.
3. Saves the policy checkpoint (`.pt`) and learning curves (JSON + PNG).
4. Evaluates GPS policy vs MPPI baseline on 10 episodes with matched seeds.
5. Renders an MP4 video of the trained policy's first evaluation episode.

### Phase 2: Hopper Environment

#### 6. `src/envs/hopper.py` — Contact-rich locomotion task

**New file.** `assets/hopper.xml` copied from Gymnasium's bundled MuJoCo assets.

Model: 6 qpos (rootx, rootz, rooty, thigh, leg, foot), 6 qvel, 3 actuators (range [-1, 1]).

Cost function mirrors the negated Gymnasium Hopper-v5 reward:
```
cost = -forward_velocity_weight * vx  -  healthy_reward * is_healthy  +  ctrl_weight * ||u||²
```

Termination: `step()` returns `done=True` when z < 0.7 or |angle| > 0.2 (hopper has fallen). During MPPI batch rollouts, there is no termination — the high cost of unhealthy states naturally discourages falling.

Observation: `qpos[1:]` (skip root x for translation invariance) + `clip(qvel, -10, 10)` = 11 dims.

#### 7. `configs/hopper_best.json` + `scripts/tuning/tune_hopper.py`

Starting MPPI hyperparameters: K=512, H=64, noise_sigma=0.3, lam=1.0, adaptive_lam=true. The tuning script uses Optuna with a GP sampler and median pruning, searching over noise_sigma and lam. Early termination during evaluation is penalised by charging the current cost for all remaining steps.

### Phase 3: Evaluation & Baselines

#### 8. `scripts/run_sb3_baseline.py` — SAC/PPO baselines

**New file.** Trains a standard model-free RL agent via stable-baselines3 on the Gymnasium version of the environment. Reports both reward (Gymnasium convention) and cost (-reward, for MPPI/GPS comparison). `stable-baselines3>=2.3.0` added to `pyproject.toml`.

#### 9. `scripts/run_ablations.py` — Ablation studies

**New file.** Runs controlled experiments varying one parameter at a time:
- **Alpha** (policy-augmented cost weight): 0.0, 0.01, 0.1, 0.5
- **K** (MPPI sample count): 128, 256, 512
- **Num conditions**: 3, 5, 10
- **Wall-clock**: policy forward pass vs MPPI planning time

Each ablation runs a full GPS training + evaluation cycle. Results are saved to `results/ablations/<env>_ablations.json`.

#### 10. `scripts/visualisation/plot_results.py` — Plotting

**New file.** Reads JSON output from run_gps.py, run_ablations.py, and run_sb3_baseline.py and generates:
- GPS training curves (cost, KL, dual variable)
- Alpha ablation bar chart with MPPI baseline reference line
- K ablation overlaid cost curves
- Num conditions bar chart
- Wall-clock comparison (policy vs MPPI ms/step)
- Cross-method comparison bar chart (GPS vs MPPI vs SAC vs PPO)

---

## File Inventory

```
NEW FILES:
  src/gps/mppi_gps.py                    Core GPS training class + KL estimators
  src/envs/hopper.py                     Hopper environment
  src/utils/evaluation.py                Shared evaluate_policy / evaluate_mppi
  assets/hopper.xml                      Hopper MuJoCo model
  configs/hopper_best.json               Hopper MPPI hyperparameters
  scripts/run_gps.py                     Main GPS training + eval script
  scripts/run_sb3_baseline.py            SAC/PPO baseline training
  scripts/run_ablations.py               Ablation study runner
  scripts/tuning/tune_hopper.py          Optuna tuning for Hopper
  scripts/visualisation/plot_results.py  Result plotting

MODIFIED FILES:
  src/envs/base.py                       Added abstract state_to_obs, obs_dim
  src/envs/mujoco_env.py                 Added default state_to_obs, obs_dim
  src/envs/acrobot.py                    Added state_to_obs, obs_dim
  src/envs/half_cheetah.py               Added state_to_obs, obs_dim
  src/envs/point_mass.py                 Added state_to_obs, obs_dim
  src/envs/gym_wrapper.py                Added state_to_obs, obs_dim
  src/utils/config.py                    Extended GPSConfig with new fields
  scripts/test_sl.py                     Updated to import from evaluation.py
  pyproject.toml                         Added stable-baselines3 dependency
  CLAUDE.md                              Updated with GPS commands + architecture
```

---

## Commands Reference

All commands assume you are in the project root with the `.venv` activated or using `uv run`:

### Install Dependencies

```bash
uv sync
```

### GPS Training

```bash
# Train GPS on acrobot (default: 50 iterations, 5 conditions, 500 steps/episode)
python -m scripts.run_gps --env acrobot

# Train GPS on hopper
python -m scripts.run_gps --env hopper

# Custom settings
python -m scripts.run_gps --env acrobot \
    --gps-iters 30 \
    --num-conditions 10 \
    --episode-length 300 \
    --alpha 0.05 \
    --seed 42

# Quick smoke test (verify everything runs end-to-end)
python -m scripts.run_gps --env acrobot \
    --gps-iters 2 \
    --num-conditions 2 \
    --episode-length 50 \
    --n-eval 3 \
    --eval-len 100
```

**Outputs:**
- `checkpoints/gps_<env>.pt` — trained policy weights
- `checkpoints/gps_<env>_curves.json` — per-iteration cost, KL, nu, loss
- `checkpoints/gps_<env>_curves.png` — learning curve plots
- `checkpoints/gps_<env>.mp4` — video of the trained policy

### MPPI Hyperparameter Tuning

```bash
# Tune MPPI for hopper (50 Optuna trials)
python -m scripts.tuning.tune_hopper

# Tune MPPI for acrobot (existing)
python -m scripts.tuning.tune_acrobot
```

**Outputs:** `configs/<env>_best.json` with tuned K, H, noise_sigma, lam.

### Standalone MPPI Control (pre-existing)

```bash
python -m scripts.run_acrobot
python -m scripts.run_cheetah
python -m scripts.run_point_mass
```

### Behaviour Cloning Pipeline (pre-existing)

```bash
# Step 1: Collect MPPI demonstrations
python -m scripts.collect_bc_demos

# Step 2: Train BC policy and compare against MPPI
python -m scripts.test_sl
```

### SB3 Baselines (SAC / PPO)

```bash
# Train SAC on Hopper-v5
python -m scripts.run_sb3_baseline --env Hopper-v5 --algo SAC --total-timesteps 500000

# Train PPO on HalfCheetah-v5
python -m scripts.run_sb3_baseline --env HalfCheetah-v5 --algo PPO --total-timesteps 1000000
```

**Outputs:** `checkpoints/sb3/<algo>_<env>` (model) + `checkpoints/sb3/<algo>_<env>_results.json`.

### Ablation Studies

```bash
# Run all ablations on acrobot (alpha, K, conditions, wall-clock)
python -m scripts.run_ablations --env acrobot --gps-iters 20

# Run on hopper
python -m scripts.run_ablations --env hopper --gps-iters 15
```

**Outputs:** `results/ablations/<env>_ablations.json`

### Plotting

```bash
# Generate all plots for acrobot
python -m scripts.visualisation.plot_results --env acrobot

# Custom directories
python -m scripts.visualisation.plot_results \
    --env hopper \
    --curves-dir checkpoints \
    --results-dir results/ablations \
    --save-dir results/plots
```

**Outputs:** `results/plots/` with PNG files for training curves, ablations, wall-clock, and method comparison.

---

## Recommended Execution Order

For a complete experiment cycle:

```bash
# 1. Tune MPPI hyperparameters for the target environment
python -m scripts.tuning.tune_hopper

# 2. Train GPS
python -m scripts.run_gps --env hopper

# 3. Train SB3 baselines for comparison
python -m scripts.run_sb3_baseline --env Hopper-v5 --algo SAC
python -m scripts.run_sb3_baseline --env Hopper-v5 --algo PPO

# 4. Run ablations
python -m scripts.run_ablations --env hopper

# 5. Generate plots
python -m scripts.visualisation.plot_results --env hopper
```

---

## Architecture Diagram

```
                         ┌─────────────────────┐
                         │   MPPIGPS.train()    │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
     │  C-step:        │  │  S-step:        │  │  BADMM update:  │
     │  MPPI planning  │  │  Policy distill │  │  Adjust nu      │
     │  with policy    │  │  via weighted   │  │  based on KL    │
     │  augmented cost │  │  MLE            │  │  vs target      │
     └────────┬────────┘  └────────┬────────┘  └─────────────────┘
              │                     │
    ┌─────────▼─────────┐ ┌────────▼────────┐
    │ make_policy_prior │ │ _distill_epoch  │
    │ α·ν·Σ log π(u|x)  │ │ mini-batch SGD  │
    │ → (K,) added to   │ │ on (obs, act,   │
    │   MPPI log-weights│ │   weights)      │
    └─────────┬─────────┘ └─────────────────┘
              │
    ┌─────────▼─────────┐
    │ MPPI.plan_step()  │
    │ K samples × H     │
    │ horizon rollouts   │
    │ via MuJoCo         │
    └───────────────────┘
```

**Data flow per GPS iteration:**

1. For each of the N initial conditions:
   - Reset env to saved state → run MPPI for T steps → collect `(obs_t, action_t)` pairs and per-timestep MPPI sample distributions `(K actions, K weights)`.
2. Concatenate all pairs across conditions → `(N×T, obs_dim)`, `(N×T, act_dim)`.
3. Train policy for `distill_epochs` passes with mini-batches of size `distill_batch_size`.
4. Compute KL between MPPI's weighted sample distribution and the updated policy (moment-matched or sample-based).
5. Adjust dual variable: `nu *= step` if KL > target, `nu /= step` if KL < target.
6. If `warm_start_policy`: roll out the policy to seed MPPI's nominal U for next iteration.
