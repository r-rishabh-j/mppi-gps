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
python -m scripts.run_hopper
python -m scripts.run_mppi

# Collect behavioral cloning demonstrations from MPPI
python -m scripts.collect_bc_demos

# Train and test supervised learning policy
python -m scripts.test_sl

# GPS training (distills MPPI into reactive policy)
# MPPI runs under a policy-augmented cost (-α·log π) then BC distills the data.
python -m scripts.run_gps --env acrobot --device auto
python -m scripts.run_gps --env hopper --gps-iters 30 --device auto
python -m scripts.run_gps --env acrobot --distill-loss mse --alpha 0.1 --device auto

# DAgger (MPPI-in-the-loop BC; policy trains on GPU, MPPI stays on CPU)
python -m scripts.run_dagger --env acrobot --dagger-iters 10 \
    --rollouts-per-iter 20 --episode-len 200 \
    --seed-from data/acrobot_bc.h5 --device auto

# Evaluate a saved policy (auto-detects env + policy class from a run dir's config.json)
python -m scripts.eval_checkpoint --ckpt experiments/dagger/<run_dir> --n-eval 10 --render

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
- **`src/policy/deterministic_policy.py`** — `DeterministicPolicy`: single-head MLP that regresses actions directly (MSE target, no `log_prob`). Used by DAgger / BC when `--deterministic` is passed. Head shape differs from `GaussianPolicy` (`act_dim` vs `2*act_dim`), so the two are not load-compatible — the wrapped checkpoint records `policy_class` so `eval_checkpoint` can auto-pick.
- **`src/policy/ema.py`** — parameter-only EMA tracker. Attached via `attach_ema(decay)`; `ema_swapped_in()` is a context manager that temporarily swaps live params with the EMA shadow (used at eval and when writing `best.pt` so the on-disk checkpoint matches the reported eval cost). Drives the `--ema-decay` / `--ema-hard-sync` / `--reset-optim-per-iter` triple in both `GPSConfig` and `DAggerConfig`.
- **`src/gps/mppi_gps.py`** — Core GPS training loop (`MPPIGPS` class). Policy-prior-only formulation: MPPI is biased by `-α·log π(u|s)` in the trajectory cost, then BC-distills the resulting (obs, action) pairs (NLL or MSE via `distill_loss`). No KL constraint, no BADMM dual updates.
- **`src/gps/mppi_gps_clip.py`** — experimental MPPI-GPS variant (clip-based update); kept beside `mppi_gps.py`. New work should default to `mppi_gps.py` unless explicitly investigating the clipped variant.
- **`src/gps/dagger.py`** — `DAggerTrainer`: roll out the current policy (with β-mixed MPPI execution), relabel every visited state via `MPPI.plan_step`, aggregate into a capped buffer, finetune the policy. MPPI stays on CPU; policy trains on the selected device.
- **`src/gps/ilqr.py`** — GPS via iLQR (skeleton/in-progress).
- **`src/utils/config.py`** — Dataclass configs (`MPPIConfig`, `PolicyConfig`, `GPSConfig`, `DAggerConfig`).
- **`src/utils/device.py`** — `pick_device("auto"|"cpu"|"mps"|"cuda")` resolves `auto` → cuda → mps → cpu.
- **`src/utils/math.py`** — Numerically stable log-sum-exp, weight computation, KL helpers (numpy).
- **`src/utils/evaluation.py`** — Shared `evaluate_policy` and `evaluate_mppi` used by GPS and BC scripts.
- **`src/utils/experiment.py`** — run-dir + checkpoint bookkeeping shared by BC, DAgger, and GPS. Defines the timestamped run-dir layout (`<base>/<YYYYmmdd-HHMMSS>_<env>_<name>/` with `config.json`, `iter_<k>.pt`, `best.pt`, `final.pt`, per-iter csv) and the wrapped-checkpoint format (`{state_dict, policy_class, round, metrics...}`). `load_checkpoint` unwraps and also accepts legacy raw state_dicts.
- **`src/utils/seeding.py`** — `seed_everything(seed)` + `add_seed_arg(parser)` helper used by training scripts.
- **`configs/`** — JSON hyperparameter configs (e.g., `acrobot_best.json`).
- **`assets/`** — MuJoCo XML model files.

## Keeping docs in sync

`commands.md` is the authoritative, user-facing cheat sheet for runnable commands
(setup, MPPI, BC, DAgger, GPS, checkpoint eval, tuning, ablations). When you add
or change a CLI flag, rename a script, or introduce a new training variant,
update `commands.md` in the same change — the brief command list in this file
stays as a high-level overview; detailed flag docs live in `commands.md`.

`AGENTS.md` covers the same scope at a higher level (project structure,
build/test, contribution style); update it together with this file when CLI or
structural changes happen.

## Key Patterns

- States are `(K, H, state_dim)` arrays, actions `(K, H, action_dim)`, costs `(K,)` — heavy NumPy broadcasting throughout.
- MPPI batch rollouts use MuJoCo's native `rollout` API with a thread pool (CPU only).
- Policy training via PyTorch runs on the device selected with `--device auto|cpu|mps|cuda` (GPS, DAgger, and BC scripts all take `--device`). MPPI always runs on CPU; only the policy crosses onto an accelerator.
- **Run-dir convention:** every training script (`run_gps`, `run_dagger`, `test_sl`) writes to a timestamped dir under `experiments/{gps,dagger,bc}/` containing `config.json` (CLI args + dataclass configs + git sha + final metrics), per-iter csv, wrapped `iter_<k>.pt` checkpoints, plus `best.pt` and `final.pt`. `eval_checkpoint --ckpt <run_dir>` auto-detects env + policy class from `config.json` and uses `best.pt`. See `src/utils/experiment.py`.
- **Two policy classes, one CLI flag:** `GaussianPolicy` (default; mu+log_sigma head) and `DeterministicPolicy` (`--deterministic`; single head, MSE-only). Head shapes differ, so a checkpoint trained under one class will not load into the other — the wrapped-checkpoint `policy_class` field disambiguates. Policies do **not** tanh-squash; bounds are applied as a clip in `act_np`. Old tanh-squashed checkpoints (with `_act_scale` / `_act_bias` buffers) are not load-compatible and need to be retrained.
- **Stabilisers (EMA + optim reset):** `--ema-decay D` shadows trainable params; eval and `best.pt` selection use the EMA snapshot. `--ema-hard-sync` promotes EMA → live weights at the end of each round (and should be paired with `--reset-optim-per-iter` to wipe stale Adam moments). Available on both GPS and DAgger.
- Python >=3.11 required. Dependency management via `uv` with `uv.lock` for reproducibility.