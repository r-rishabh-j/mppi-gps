# MPPI-GPS Implementation Report

## Overview

This document describes the implementation of an MPPI-based **Guided Policy Search (GPS)** system that distills an online MPPI trajectory optimiser into a cheap, reactive neural-network policy. The core idea: MPPI produces excellent trajectories at high computational cost (K rollouts per step); GPS trains a feedforward policy to mimic MPPI's behaviour, enabling real-time deployment without the simulator-in-the-loop.

Two distillation methods are implemented and share the same student (`GaussianPolicy`) and evaluation harness:

- **MPPI-GPS** (`src/gps/mppi_gps.py`) — policy-augmented MPPI cost + weighted MLE, optionally with BADMM-style KL constraint. Supports a "policy-prior-only" variant (`--disable-kl`) which is used by default for current experiments.
- **DAgger** (`src/gps/dagger.py`) — MPPI-in-the-loop BC with a linearly-decayed mixing coefficient β and an aggregating buffer.

A standalone BC pipeline (`scripts/collect_bc_demos.py` → `scripts/test_sl.py`) is kept as a one-shot baseline.

All MPPI batch rollouts run on CPU via MuJoCo's native `rollout` API with a thread pool. The student policy can train on CPU / MPS / CUDA — selected with `--device auto|cpu|mps|cuda` — resolved by `src/utils/device.py::pick_device`. The MJX / JAX GPU path that earlier drafts carried has been **removed**; `--backend` no longer exists.

---

## Current Architecture

### Student Policy — `src/policy/gaussian_policy.py`

Diagonal-Gaussian MLP with three training-stability features, all gated by `PolicyConfig` and saved into `state_dict` via registered buffers (so checkpoints restore cleanly without any custom load code):

1. **Running observation normalizer** (`RunningNormalizer`). Non-trainable buffers `mean`, `var`, `count` updated via a Welford parallel merge. Critically, `update()` is **explicit** — only the supervised training path (`train_weighted`) calls it, so MPPI's policy-prior queries (which pass trajectory-sample actions through `log_prob_np`) cannot corrupt the running stats.
2. **`log_sigma` clamp** in `_head()` to `[log_sigma_min, log_sigma_max]` (defaults `[-5.0, 2.0]`). Prevents the classic NLL pathologies of `σ → 0` (infinite log-likelihood gradients on slight action mismatches) and `σ → ∞` (vanishing signal).
3. **Optional tanh-squashed head** (`squash_tanh=True`). Rescales via buffers `_act_scale` / `_act_bias` to the env action box, with a change-of-variables correction `Σ_i log(1 - tanh² u_i + ε) + Σ_i log(scale_i)` applied in `log_prob`. Off by default for acrobot/point-mass; intended for hopper/cheetah where MPPI teacher samples live on the action-box boundary.

`train_weighted` runs one Adam step of weighted NLL. `act_np` / `log_prob_np` are the numpy bridges used by MPPI and the eval harness.

### MPPI-GPS core — `src/gps/mppi_gps.py`

**`make_policy_prior(policy, env, alpha, nu)`** — builds the callable plugged into MPPI's `prior` arg. Returns `+α·ν · Σ_t log π(u_t | obs_t)` per trajectory. Because MPPI computes `log_w = -S/λ + log_prior`, a positive prior reduces the effective cost of policy-likely trajectories.

**KL estimators**:
- **`compute_kl_moment_matched`** (Eq. 3) — fits `N(μ_t, Σ_t)` to the K weighted MPPI first-step actions, then computes closed-form `KL(p || q)` against the diagonal-Gaussian policy at `obs_t`.
- **`compute_kl_sample_based`** (Eq. 4) — `KL ≈ Σ_k w_k (log w_k - log π(u_k | x_k))` directly on weighted particles; higher variance but multi-modal-aware.

**`MPPIGPS.train(num_iterations=None, checkpoint_dir=None, env_tag="gps")`** — main loop:

- **C-step**: for each of the `num_conditions` fixed initial states, run MPPI for `episode_length` steps with the policy-augmented prior. Collect `(obs_t, action_t)` pairs and — unless `disable_kl_constraint` — per-timestep `(K actions, K weights)` for KL.
- **S-step**: concatenate all conditions' data, run `distill_epochs` passes of mini-batched distillation (`distill_batch_size`). `distill_loss` selects `"nll"` (weighted NLL via `train_weighted`) or `"mse"` (MSE on policy mean — reduces to plain BC).
- **KL + BADMM**: if enabled, compute mean KL across conditions and update ν multiplicatively (`ν *= step` if KL > target, else `ν /= step`, clamped to `[1e-4, 1e4]`).
- **Checkpointing** (new): per iteration, write `{env_tag}_iter{k:03d}.pt`. Track the best-so-far `mean_cost`; on improvement, overwrite `{env_tag}_best.pt` and record `(best_iter, best_cost)` on the returned `GPSHistory`.
- **Warm-start**: iter ≥ 1 seeds MPPI's nominal `U` by rolling the current policy from the init state (`_warm_start_mppi`). Terminal-done states are padded with zero actions so post-terminal garbage obs never reach the policy.

**Policy-prior-only variant** (`disable_kl_constraint=True` / `--disable-kl`) — skips KL / BADMM entirely and holds `ν` constant at `badmm_init_nu`. The C-step's prior weight is the constant `α·ν`; the S-step is plain supervised distillation.

**Key correctness fixes carried in the current implementation**:
- Iteration-0 always calls `self.mppi.reset()` (zeros the nominal `U`); earlier drafts skipped the reset when `warm_start_policy=True`, leaking MPPI state across conditions.
- KL first-step action / weight buffers are `np.array(...)` copies of `rollout_data` — MPPI overwrites `_last_*` slots on the next `plan_step`, so views would silently alias.
- Warm-start respects env `done`, padding with zero actions.

### DAgger — `src/gps/dagger.py`

`DAggerTrainer` runs the policy in the env and relabels every visited full state with `mppi.plan_step` as the expert. β-schedule (`linear` 1→0 over K/2 iters, or `constant_zero`) mixes policy vs MPPI **execution** during rollout; relabelling always uses MPPI. An aggregating buffer (`buffer_cap`, FIFO eviction) persists across iterations. Per iter: finetune via MSE-on-mean for `distill_epochs` on the full buffer, optionally seeded from a BC h5 via `--seed-from`.

### Shared components

- **`src/utils/config.py`** — four dataclasses: `MPPIConfig` (K / H / lam / noise_sigma / adaptive_lam / n_eff_threshold, `load(env_name)` reads `configs/<env>_best.json`), `PolicyConfig` (hidden_dims, lr, activation, plus the four new stability fields `obs_norm`, `log_sigma_min`, `log_sigma_max`, `squash_tanh`), `GPSConfig`, `DAggerConfig`.
- **`src/utils/evaluation.py`** — `evaluate_policy` and `evaluate_mppi` using a shared seed schedule `(seed + ep)` so GPS-vs-MPPI per-episode comparisons start from identical initial conditions.
- **`src/utils/device.py`** — `pick_device("auto"|"cpu"|"mps"|"cuda")` resolving `auto` → cuda → mps → cpu.
- **`src/utils/math.py`** — log-sum-exp, weight computation, weighted-mean-cov, KL helpers (numpy).
- **`src/envs/`** — `BaseEnv` + `MuJoCoEnv` base, concrete envs: `acrobot`, `half_cheetah`, `point_mass`, `hopper`. Factory: `src.envs.make_env(name)`. Every env exposes `state_to_obs(states)` for the policy-sized projection and `obs_dim` as a property. Hopper terminates via `done=True` on `z < 0.7` or `|angle| > 0.2` in single-step mode; batch rollouts don't terminate — high unhealthy cost discourages falling.

---

## Entry Points

### MPPI (open-loop, baseline)

```bash
python -m scripts.run_acrobot       # per-env tuning-style harness
python -m scripts.run_cheetah
python -m scripts.run_hopper
python -m scripts.run_point_mass
python -m scripts.run_mppi          # generic entry
```

### Behaviour cloning (one-shot)

```bash
python -m scripts.collect_bc_demos   # MPPI → data/acrobot_bc.h5
python -m scripts.test_sl            # MSE on policy mean vs MPPI baseline
```

### DAgger

```bash
python -m scripts.run_dagger --env acrobot \
    --dagger-iters 10 --rollouts-per-iter 20 --episode-len 200 \
    --seed-from data/acrobot_bc.h5 --device auto
```

Outputs: per-iter `checkpoints/dagger/dagger_<env>_iter<k>.pt`, metrics csv at `results/dagger_<env>/dagger_log.csv`.

### GPS

```bash
# KL-constrained GPS
python -m scripts.run_gps --env acrobot --device auto
python -m scripts.run_gps --env hopper --gps-iters 30 --device auto

# Policy-prior-only (no KL / no BADMM)
python -m scripts.run_gps --env acrobot --disable-kl --distill-loss nll \
    --alpha 0.1 --nu 1.0 --device auto --gps-iters 20
```

Relevant CLI flags: `--disable-kl`, `--distill-loss {nll,mse}`, `--alpha`, `--nu`, `--gps-iters`, `--num-conditions`, `--episode-length`, `--n-eval`, `--eval-len`, `--ckpt-dir`, `--device`, `--seed`.

**Outputs** (under `--ckpt-dir`, default `checkpoints/`):
- `gps_<env>_iter<k:03d>.pt` — per-iteration policy state_dict (normalizer + squash buffers included).
- `gps_<env>_best.pt` — state_dict from the iteration with the lowest mean rollout cost.
- `gps_<env>.pt` — final-iter state_dict.
- `gps_<env>_curves.json` / `.png` — cost / KL / ν / distill-loss curves.
- `gps_<env>.mp4` — first eval-episode video of the trained policy.

### Evaluate an existing checkpoint

```bash
python -m scripts.eval_checkpoint --env acrobot \
    --ckpt checkpoints/gps_acrobot_best.pt \
    --n-eval 10 --render
```

Loads any compatible `state_dict` (GPS / DAgger / BC) into a fresh `GaussianPolicy(obs_dim, act_dim, PolicyConfig())`; normalizer stats and tanh-squash buffers restore automatically because they're registered buffers. Caveat: the script hardcodes `PolicyConfig()` defaults, so a checkpoint trained with non-default `squash_tanh` will fail `strict` load until a matching flag is plumbed.

### Tuning / baselines / ablations

```bash
python -m scripts.tuning.tune_acrobot
python -m scripts.tuning.tune_cheetah
python -m scripts.tuning.tune_hopper
python -m scripts.run_sb3_baseline --env Hopper-v5 --algo SAC
python -m scripts.run_ablations --env acrobot
python -m scripts.visualisation.plot_results --env acrobot
```

Tuning uses Optuna (GP sampler, median pruning) and writes `configs/<env>_best.json` with `K`, `H`, `noise_sigma`, `lam`. `MPPIConfig.load(env_name)` consumes this; fields not in the JSON fall back to the dataclass defaults (`adaptive_lam=False`, `n_eff_threshold=64.0`).

---

## Verified Training-Stability Fixes

Smoke test on acrobot (`--disable-kl --distill-loss nll --alpha 0.1 --nu 1.0 --gps-iters 3 --num-conditions 2 --episode-length 80 --device cpu`):

- Per-iter and best-of-run checkpoints are both written; `best_iter` / `best_cost` reported at the end of `run_gps.py`.
- Loading `gps_acrobot_iter002.pt` into a fresh policy shows `normalizer.count = 2400` and `|mean|_max ≈ 2.1`, `var ∈ [1.10, 34.04]` — stats have clearly shifted from the `mean=0, var=1` init, confirming the explicit-update wiring works end-to-end and survives `state_dict` round-trip.
- `log_sigma` on 100 random-reset acrobot obs ranges in `[−2.41, −0.69]`, well inside the `[−5.0, 2.0]` clamp.
- Distillation loss decreased monotonically across iters (acrobot smoke: 0.81 → 0.53 → 0.39).

---

## File Inventory

```
src/gps/
  mppi_gps.py                            MPPI-GPS trainer, KL estimators, BADMM, checkpointing
  dagger.py                              DAgger trainer (MPPI-in-the-loop BC)
  ilqr.py                                iLQR skeleton (in progress)

src/policy/
  gaussian_policy.py                     Diagonal-Gaussian MLP +
                                         RunningNormalizer, log_sigma clamp, optional tanh squash

src/envs/
  base.py  mujoco_env.py  gym_wrapper.py
  acrobot.py  half_cheetah.py  point_mass.py  hopper.py
  __init__.py                            make_env(name) factory

src/utils/
  config.py                              MPPIConfig, PolicyConfig, GPSConfig, DAggerConfig
  device.py                              pick_device helper
  evaluation.py                          evaluate_policy, evaluate_mppi
  math.py                                weights / KL / weighted moments

scripts/
  run_gps.py                             GPS entry (KL + policy-prior-only)
  run_dagger.py                          DAgger entry
  eval_checkpoint.py                     Standalone checkpoint evaluator
  run_mppi.py  run_acrobot.py  run_cheetah.py  run_hopper.py  run_point_mass.py
  collect_bc_demos.py  test_sl.py        One-shot BC pipeline
  run_sb3_baseline.py                    SAC / PPO baselines
  run_ablations.py                       Ablation runner
  tuning/tune_{acrobot,cheetah,hopper}.py
  visualisation/{plot_results,plot_cost,visualise_rollouts}.py

configs/
  acrobot_best.json                      K=256, H=256, noise_sigma=0.057, lam=0.005
  hopper_best.json                       K=512, H=64, noise_sigma=0.3, lam=1.0, adaptive_lam=true

assets/
  acrobot.xml  half_cheetah.xml  hopper.xml  point_mass.xml
```

---

## Data Flow (GPS, per iteration)

```
                         ┌──────────────────────┐
                         │   MPPIGPS.train()    │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
     │  C-step:        │  │  S-step:        │  │  BADMM update   │
     │  MPPI with      │  │  Distill via    │  │  (skipped when  │
     │  policy prior   │  │  NLL or MSE     │  │  --disable-kl)  │
     └────────┬────────┘  └────────┬────────┘  └─────────────────┘
              │                     │
    ┌─────────▼─────────┐ ┌────────▼────────┐
    │ make_policy_prior │ │ _distill_epoch  │
    │ +α·ν·Σ log π(u|x) │ │ mini-batch SGD  │
    │ added to MPPI     │ │ (weighted NLL   │
    │ log-weights       │ │  or MSE)        │
    └─────────┬─────────┘ └─────────────────┘
              │
    ┌─────────▼─────────┐        ┌──────────────────────┐
    │ MPPI.plan_step()  │───────▶│ _warm_start_mppi()   │
    │ K×H MuJoCo rollout│        │ seeds next-iter U    │
    └───────────────────┘        │ from policy rollout  │
                                 └──────────────────────┘
```

Per iteration:
1. For each of the `num_conditions` fixed init states: reset env → run MPPI + prior for `episode_length` steps → collect `(obs_t, action_t)` and optionally the K-sample first-step distribution.
2. Concatenate `(N×T, obs_dim)` / `(N×T, act_dim)` → `distill_epochs` mini-batched passes.
3. Save `{env_tag}_iter{k:03d}.pt`; if `mean_cost` improved, overwrite `{env_tag}_best.pt`.
4. If KL enabled: compute moment-matched or sample-based KL, update ν.
5. If `warm_start_policy`: roll policy from each init state to seed next iter's MPPI `U`.

---

## Deliberate Non-Goals

Tracked explicitly to keep scope tight:
- GMM / diffusion / multimodal policy heads — deferred. A unimodal diagonal Gaussian is the current student.
- Action chunking / temporal context on the policy — deferred.
- Removing the dead `weights` parameter on `train_weighted` — deferred cleanup. All current callers pass uniform weights.
- Revisiting the BADMM / KL path beyond whatever's needed for `PolicyConfig` compatibility.
- Re-introducing the MJX / JAX GPU pipeline.
