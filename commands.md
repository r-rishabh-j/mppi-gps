# Commands

## Setup

```bash
uv sync
```

## MPPI (open-loop control)

```bash
python -m scripts.run_point_mass
python -m scripts.run_acrobot
python -m scripts.run_cheetah
python -m scripts.run_hopper
python -m scripts.run_adroit_pen     # 24-DoF Shadow Hand pen reorientation
python -m scripts.run_adroit_relocate # 30-DoF Adroit arm+hand pick-and-place
python -m scripts.run_mppi           # generic entry
```

## GPU rollouts via Warp (`--warp`)

`adroit_relocate` has a `mujoco_warp`-backed batch rollout that runs MPPI's
K trajectory rollouts on GPU instead of the CPU thread pool. The whole
`H`-step rollout is captured into a CUDA graph on the first call and replayed
thereafter — that's where the speedup comes from. Same cost / observations /
sensors as the CPU path; only `batch_rollout` is replaced.

**Install** (NVIDIA GPU + CUDA required for the graph-replay speedup):
```bash
uv pip install warp-lang mujoco-warp
```

**Standalone MPPI run:**
```bash
python -m scripts.run_adroit_relocate --warp                # opt-in flag
```

**GPS / DAgger:** the GPS script gates everything behind a single flag and
pins `nworld = MPPIConfig.K` automatically:
```bash
python -m scripts.run_gps --env adroit_relocate --auto-reset \
    --warp --gps-iters 40 --device auto --policy-prior mean_distance
```

Files:
- `src/envs/warp_rollout.py` — `WarpRolloutMixin` (general; supports mocap).
- `src/envs/adroit_relocate_warp.py` — `AdroitRelocateWarp(WarpRolloutMixin, AdroitRelocate)`.
- `src/envs/__init__.py` — `make_env(name, use_warp=True, nworld=K)` dispatches.

Constraints:
- Only `adroit_relocate` has a Warp variant currently. Other envs raise
  on `--warp`. Adding one for any `na==0` env is a ~10-line subclass.
- `nworld` is **fixed at construction**. Changing MPPI's `K` (e.g. via a
  config edit) means re-running the script — re-instantiation handles it.
- Graph replay is CUDA-specific. macOS / non-NVIDIA hosts should stick
  with the default CPU path; the code lazy-imports `warp` so the CPU
  path doesn't need it installed.
- The Warp path doesn't track sim time (`state[..., 0] = 0`). No cost or
  obs function in this codebase reads time, so this is invisible — but
  any future code that does will need to be aware.

## Behavior cloning (one-shot)

Two-step pipeline: **collect** MPPI demos to `data/<env>_bc.h5` (cached —
re-running is a no-op unless `--force`), then **train** a policy on that
h5 into a run dir under `experiments/bc/<timestamp>_<env>_<name>/`
(config.json, per-epoch csv, best/final/iter checkpoints, loss curve, mp4).

```bash
# Acrobot — Gaussian policy (MSE on mean)
python -m scripts.collect_bc_demos --env acrobot
python -m scripts.test_sl --env acrobot --device auto

# Hopper — terminating env → --auto-reset on collection; deterministic policy
python -m scripts.collect_bc_demos --env hopper --auto-reset -M 30 -T 500
python -m scripts.test_sl --env hopper --deterministic --device auto

# Re-collect a cached dataset after an env / reward change
python -m scripts.collect_bc_demos --env acrobot --force

# Warm-start from any wrapped BC / DAgger / GPS checkpoint
python -m scripts.test_sl --env acrobot --device auto \
    --init-ckpt experiments/bc/<run>/best.pt --num-epochs 30

# Fast smoke (CPU, ~1 min)
python -m scripts.test_sl --env acrobot --num-epochs 5 \
    --eval-every 0 --n-eval-eps 3 --eval-ep-len 150 --device cpu
```

Key flags (run with `--help` for the full list):
- **collect_bc_demos** — `--env`, `-M / --num-trajectories`, `-T / --trajectory-length`,
  `--out`, `--force`, `--auto-reset`, `--seed`.
- **test_sl** — `--env`, `--device`, `--deterministic`, `--init-ckpt`, `--demos`,
  `--num-epochs`, `--batch-size`, `--val-frac`, `--eval-every`, `--ckpt-every`,
  `--n-eval-eps`, `--eval-ep-len`, `--exp-name`, `--exp-dir`.

Run-dir contents: `config.json`, `bc_log.csv`, `loss.png`, `<env>.mp4`,
wrapped `iter_<epoch>.pt` checkpoints, plus `best.pt` (best-val-mse) and `final.pt`.

## DAgger (MPPI-in-the-loop BC)

Iteratively rolls the policy, relabels visited states with MPPI, and finetunes.
MPPI runs on CPU; policy training uses the selected device. Works with either
a stochastic `GaussianPolicy` (default; trained on MSE-of-mean, log_sigma head
unused) or a `DeterministicPolicy` (direct action regression, via `--deterministic`).

```bash
# full run on acrobot, seeded from the BC dataset if present
python -m scripts.run_dagger --env acrobot \
    --dagger-iters 10 --rollouts-per-iter 20 --episode-len 200 \
    --seed-from data/acrobot_bc.h5 --device auto

# resume DAgger (or start from a GPS / BC / prior DAgger checkpoint)
python -m scripts.run_dagger --env acrobot --deterministic \
    --init-ckpt checkpoints/dagger/dagger_acrobot_iter09.pt \
    --dagger-iters 5 --rollouts-per-iter 20 --episode-len 200 \
    --device auto

# deterministic policy + BC warmup before the DAgger loop
python -m scripts.run_dagger --env acrobot --deterministic \
    --warmup-rollouts 20 --warmup-epochs 30 \
    --dagger-iters 10 --rollouts-per-iter 20 --episode-len 200 \
    --beta-schedule linear --device auto

# hopper (terminating env — use --auto-reset so early falls don't
# waste the per-rollout step budget; env is resolved via the registry)
python -m scripts.run_dagger --env hopper --deterministic --auto-reset \
    --warmup-rollouts 30 --warmup-epochs 50 \
    --dagger-iters 30 --rollouts-per-iter 10 --episode-len 500 \
    --beta-schedule linear --distill-epochs 6 --buffer-cap 50000 --device cuda

# quick smoke test (~1–2 min)
python -m scripts.run_dagger --env acrobot \
    --dagger-iters 2 --rollouts-per-iter 3 --episode-len 80 \
    --distill-epochs 3 --n-eval-eps 3 --eval-ep-len 150 --device auto
```

Flags:
- `--env {acrobot,adroit_pen,adroit_relocate,half_cheetah,hopper,point_mass}` — resolved via the `src.envs.make_env` registry.
- `--device auto|cpu|mps|cuda` — auto resolves to cuda → mps → cpu.
- `--deterministic` — use `DeterministicPolicy` (single-head, direct action regression) instead of `GaussianPolicy`.
- `--beta-schedule linear|constant_zero` — β decays 1→0 over K/2 iters, or always 0.
- `--buffer-cap 200000` — aggregated buffer capacity (oldest evicted).
- `--auto-reset` — on episode termination during a rollout slot, reset env + MPPI and keep relabeling until the slot's step budget is spent. Recommended for terminating envs like hopper; off by default so acrobot / non-terminating behavior is unchanged.
- `--init-ckpt PATH` — load policy weights before warmup / DAgger. The checkpoint class must match `--deterministic` (deterministic vs Gaussian head shapes differ).
- `--seed-from PATH` — warm-start buffer from an existing BC h5 (no pre-training, just seeded rows).
- `--warmup-rollouts N` — collect N pure-MPPI rollouts **and BC-train the policy on them** before the DAgger loop. Rows land in the aggregate buffer with `round_idx=-1`.
- `--warmup-epochs E` — epochs of BC pre-training on warmup rollouts (default 20; ignored when `--warmup-rollouts=0`).
- `--warmup-cache PATH` — optional h5 path for warmup rollouts. If the file exists it's loaded (skipping the MPPI collection); otherwise rollouts are collected and saved there for the next run. **Off by default** — pass a path only when you want to persist. Compatible with `collect_bc_demos` output: point this at an existing `data/<env>_bc.h5` to skip warmup collection. Delete the file to invalidate a stale cache.
- `--exp-name NAME` — human-readable experiment name (default `run`). Appended to the run dir.
- `--exp-dir PATH` — parent dir under which a `<timestamp>_<env>_<name>/` run dir is created. Default `experiments/dagger`.
- `--ema-decay D` — exponential moving average decay over policy trainable params (e.g. `0.999`). 0/unset disables. Eval and `best.pt` selection use the EMA snapshot so the checkpoint on disk matches the reported cost.
- `--ema-hard-sync` — promote EMA → live weights at end of each DAgger round. No effect without `--ema-decay > 0`. Pair with `--reset-optim-per-iter`.
- `--reset-optim-per-iter` — wipe Adam m/v moments at end of each round. Required for Adam consistency after a hard-sync.

Outputs (all inside the run dir `experiments/dagger/<timestamp>_<env>_<name>/`):
- `config.json` — CLI args, DAgger/Policy/MPPI configs, env, device, git sha, start/end time, MPPI baseline, best iter + cost. Written at start, updated at end.
- `iter_<k>.pt` — per-iteration wrapped checkpoint: `{state_dict, policy_class, round, train_mse, val_mse, eval_mean_cost, eval_std_cost}`.
- `best.pt` / `final.pt` — copies of the best-by-eval and last iterations.
- `dagger_log.csv` — per-iter metrics + MPPI baseline (updated incrementally).

## GPS (KL-constrained distillation)

Each run creates a timestamped dir `experiments/gps/<YYYYmmdd-HHMMSS>_<env>_<name>/`
(override parent via `--exp-dir`, name via `--exp-name`).

```bash
python -m scripts.run_gps --env acrobot --device auto
python -m scripts.run_gps --env hopper --auto-reset --gps-iters 30 --device auto

# warm-start from a DAgger / BC checkpoint before the GPS loop
python -m scripts.run_gps --env acrobot --exp-name warmstart \
    --init-ckpt checkpoints/dagger/<run>/best.pt --device auto

# hopper — terminating env, needs auto-reset so dataset size doesn't collapse on early iters
python -m scripts.run_gps --env hopper --auto-reset \
    --distill-loss mse --alpha 0.1 \
    --gps-iters 30 --episode-length 500 --device auto
```

Outputs (inside the run dir):
- `iter_<k:03d>.pt` — per-iteration wrapped checkpoints (state_dict + `policy_class` + metrics).
- `best.pt` — copy of the iteration with the lowest mean rollout cost.
- `final.pt` — copy of the last iteration's weights.
- `config.json` — CLI args, all dataclass configs, policy class, git sha, eval summary.
- `curves.json` / `curves.png` — cost / distill-loss curves.
- `<env>.mp4` — first eval-episode video of the trained policy.

### Common flags

GPS runs MPPI under a policy-augmented cost (`S = S_env + ... - α·log π(u|s)`)
then BC-distills the resulting (obs, action) pairs. No KL constraint, no BADMM
dual updates.

```bash
python -m scripts.run_gps --env acrobot --distill-loss nll \
    --alpha 0.1 --device auto --gps-iters 20

# quick smoke test
python -m scripts.run_gps --env acrobot --distill-loss nll \
    --alpha 0.1 --gps-iters 3 --num-conditions 2 --episode-length 80 \
    --n-eval 2 --eval-len 80 --device cpu
```

Flags:
- `--distill-loss {nll,mse}` — `mse` matches plain BC, `nll` uses weighted NLL (default).
- `--alpha` — weight on `-log π` inside the MPPI cost (policy-prior strength).
- `--init-ckpt PATH` — load a wrapped or raw state_dict into the policy before the GPS loop (warm-start from BC / DAgger / a previous GPS run).
- `--auto-reset` — on env termination during the C-step, reset to a fresh random init and keep collecting until `episode_length` steps are taken. Recommended for hopper and any other env that can terminate early; without it, early iters produce tiny distillation datasets.
- `--warm-start-policy` — (advanced) seed MPPI's nominal `U` from a policy rollout before each condition. Off by default — the `-α·log π` prior already biases MPPI toward the policy, and on terminating envs the zero-padding past `done` actively hurts.
- `--distill-buffer-cap N` — keep the last `N` (obs, action) rows across GPS iterations and distill from that buffer. 0/unset = current per-iter behaviour. Sub-episodes split at each `done` (variable-length with `--auto-reset`); FIFO eviction by oldest whole episode until total rows ≤ cap.
- `--ema-decay D` — exponential moving average decay over the policy's trainable params (e.g. `0.999`). 0/unset disables. Eval and `best.pt`/`final.pt` reflect the EMA snapshot, so the checkpoint matches the reported eval cost. Without `--ema-hard-sync` the shadow stays passive (MPPI's prior uses fresh training weights).
- `--ema-hard-sync` — at end of each S-step, promote the EMA shadow to live weights (θ ← EMA). Next iter's MPPI prior + S-step start from the smoothed policy. No effect without `--ema-decay > 0`. Strongly recommended to pair with `--reset-optim-per-iter`.
- `--reset-optim-per-iter` — rebuild the policy's Adam optimizer at end of each GPS iter, wiping m/v moments. Required for Adam-state consistency after a hard-sync; also useful on its own because each GPS iter is a new supervised task (shifting C-step data) and stale momentum can blow through the `--prev-iter-kl-coef` trust region.
- `--prev-iter-kl-coef C` — trust-region KL penalty `C·E_obs[KL(π_θ‖π_prev)]` added to the S-step distill loss. Stabilises iter-to-iter drift when C-step data shifts. NLL mode only (silently ignored for `--distill-loss mse`).
- `--dagger-relabel` — decouple executor from teacher in the C-step: run MPPI twice per timestep — once WITH the policy prior (executor, steers the env) and once WITHOUT (side-effect-free `dry_run`, its action is the training label). Breaks the self-reinforcing loop where the executed MPPI action is already tilted toward the current policy. No-op when `α == 0`. Works under any `open_loop_steps`, but with `open_loop_steps > 1` the label call forces a full rollout every timestep (cached chunk actions are prior-biased, not valid labels) — you lose the open-loop wallclock speedup on the label side.
- `--gps-iters`, `--num-conditions`, `--episode-length` — override `GPSConfig` defaults.
- `--n-eval`, `--eval-len` — evaluation episode count / length.
- `--exp-name`, `--exp-dir` — run-dir naming (parent defaults to `experiments/gps`).
- `--device auto|cpu|mps|cuda` — policy training device (MPPI always runs on CPU).

### α tuning: schedule and KL-adaptive

`--alpha` controls the policy-prior weight in MPPI's cost. Three modes for choosing it per-iter, in order of sophistication:

**Mode 1 — constant** (default). `α = --alpha` every iter.

```bash
python -m scripts.run_gps --env adroit_relocate --alpha 0.1 --gps-iters 40 --device auto
```

**Mode 2 — non-linear schedule.** Ramp α from `--alpha-start` to `--alpha` over `--alpha-warmup-iters`, then plateau. Useful at iter 0 when the policy is untrained: starting α at 0 lets MPPI explore freely while the policy bootstraps, ramping the prior in later for on-policy state coverage.

```bash
python -m scripts.run_gps --env adroit_relocate --auto-reset \
    --alpha 0.1 \
    --alpha-schedule smoothstep \    # or linear, cosine, constant
    --alpha-warmup-iters 8 \         # ramp 0 → 0.1 over the first 8 iters
    --alpha-start 0.0 \
    --gps-iters 40 --device auto
```

| flag | meaning |
|---|---|
| `--alpha-schedule {constant,linear,smoothstep,cosine}` | shape of the ramp; `constant` (default) disables the schedule, `smoothstep` is the recommended ramp (cubic, derivative 0 at endpoints) |
| `--alpha-warmup-iters N` | iters over which α ramps from `--alpha-start` to `--alpha` |
| `--alpha-start S` | starting α (default 0.0) |

**Mode 3 — KL-adaptive α with the unbiased teacher operand** (Gaussian policies only). `α` becomes a *dual variable* auto-adjusted each iter to track a target KL between MPPI's **unbiased** teacher distribution (cost-only posterior, prior stripped from the importance weights) and the global policy:

```
kl_est = E_state[ KL(N(μ_p(s), σ_p²(s)) ‖ π_θ(·|s)) ]      # μ_p, σ_p² re-weighted by S − track
α ← α / kl_step_rate    if kl_est > kl_target  (policy is bad; let MPPI explore)
α ← α · kl_step_rate    if kl_est < kl_target  (policy matches teacher; lean on it)
```

Note the **inverted update direction** vs vanilla MDGPS: with the unbiased operand, large `kl_est` means "policy is far from the cost-optimal teacher" → we want MPPI to ignore the prior and explore. With the biased operand the direction would flip, but that variant has a degenerate fixed point (biased MPPI → π_θ by construction as α → ∞, so KL → 0 mechanically regardless of policy quality). Stripping the prior breaks the self-reference and gives a meaningful "policy bad → small α → escape" loop.

```bash
# Adroit (30-D) — target ≈ act_dim × 0.05 ≈ 1.5
python -m scripts.run_gps --env adroit_relocate --auto-reset \
    --kl-target 1.5 \                # per-state KL summed over action dims
    --alpha 0.05 \                   # seeds α at end of warmup
    --alpha-warmup-iters 5 \         # adaptive rule activates at iter 5
    --alpha-schedule smoothstep --alpha-start 0 \   # warmup ramp shape
    --gps-iters 40 --device auto --policy-prior mean_distance

# Low-D toy (acrobot 1-D, hopper 3-D) — much smaller target
python -m scripts.run_gps --env acrobot --kl-target 0.1 \
    --alpha 0.05 --alpha-warmup-iters 3 --gps-iters 30 --device auto
```

| flag | default | meaning |
|---|---|---|
| `--kl-target T` | 0.0 (off) | per-state KL target **summed over action dims** — scale with act_dim. Rule of thumb `act_dim × 0.05`: ~0.1 for acrobot (2-D), ~0.3 for cheetah (6-D), ~1.5 for adroit_relocate (30-D). Set > 0 to enable adaptive mode |
| `--kl-alpha-min` | 0.001 | lower bound on the dual α (allows multiplicative escape) |
| `--kl-alpha-max` | 0.5 | upper bound on the dual α. α ≫ 0.1 typically crushes MPPI's exploration regardless of the constraint, so growing past 0.5 is rarely productive |
| `--kl-step-rate` | 1.5 | multiplicative update rate per iter; 2.0+ for faster escape from sticky regimes |
| `--kl-sigma-floor-frac` | 0.5 | floor on σ_p (local-policy std) in the KL estimator, expressed as a fraction of MPPI's proposal std. Prevents `log(σ_θ/σ_p)` from exploding when MPPI's softmin concentrates onto a few samples (var_p → 0). 0 disables the floor (legacy: clamp at 1e-6, biases kl_est upward by tens of nats per state) |

Notes:
- **Per-state KL is summed over action dims** — `kl_target=0.1` is meaningless on 30-D Adroit (would need `kl_est` ≈ 0.003 per dim, which never happens in practice with MPPI's softmin). Use the per-dim rule of thumb above.
- The schedule (mode 2) is used **during the warmup window** — `--alpha-warmup-iters` iterations of standard schedule behavior. After warmup, the KL-adaptive rule takes over and the schedule fields become inert.
- KL-adaptive is **Gaussian-only** (needs σ for the closed-form Gaussian KL). Silently falls back to schedule under `--deterministic`.
- Per-iter `α`, `kl_est`, target are printed to the tqdm postfix and the per-iter line so you can watch the dual variable adapt.
- **If `kl_est` is stuck at hundreds of nats and α saturates at the cap**, the σ_p collapse is dominating — make sure `--kl-sigma-floor-frac` is at its default 0.5 (not 0). With the floor active, `kl_est` for a 30-D env should land in the 1–30 range when the policy and MPPI are reasonably aligned.

## Evaluate a saved checkpoint

Load any `.pt` policy state_dict and evaluate it. `--ckpt` accepts either a
single `.pt` file or a run dir from `scripts.run_dagger` — in the latter case
`best.pt` is used and `--env` / `--deterministic` are auto-detected from the
run's `config.json`.

Policies no longer tanh-squash — the network output is unconstrained and
bounds are applied as a clip at `act_np`. Checkpoints saved under the old
tanh-squash scheme (with `_act_scale` / `_act_bias` buffers) are not
load-compatible with the current policy and need to be retrained.

```bash
# eval the best checkpoint from a DAgger run dir (env + policy auto-detected)
python -m scripts.eval_checkpoint \
    --ckpt experiments/dagger/20260418-162125_acrobot_smoke \
    --n-eval 10 --render

# legacy GPS checkpoint (raw state_dict)
python -m scripts.eval_checkpoint --env acrobot \
    --ckpt checkpoints/gps_acrobot_best.pt \
    --n-eval 10 --render

# deterministic policy from a specific iter .pt inside a run dir
python -m scripts.eval_checkpoint --deterministic \
    --ckpt experiments/dagger/20260418-162125_acrobot_smoke/iter_09.pt \
    --n-eval 10 --render
```

Flags:
- `--ckpt` — path to a policy `state_dict` (GPS / DAgger / BC all compatible).
- `--deterministic` — load weights into `DeterministicPolicy`. Must match the class the checkpoint was trained under (shapes differ: deterministic head has `act_dim` outputs, Gaussian head has `2*act_dim`).
- `--n-eval`, `--eval-len` — evaluation episode count / length.
- `--render` — save an mp4 of episode 0 next to the checkpoint (or at `--video-out`).
- `--device auto|cpu|mps|cuda` — inference device.

## Baselines, tuning, ablations

```bash
python -m scripts.run_sb3_baseline --env Hopper-v5 --algo SAC

python -m scripts.tuning.tune_acrobot
python -m scripts.tuning.tune_cheetah
python -m scripts.tuning.tune_hopper

python -m scripts.run_ablations --env acrobot
python -m scripts.visualisation.plot_results --env acrobot
```


### Bookeeping for experiments

#### Hopper MPPI GPS
python -m scripts.run_gps --env hopper --auto-reset --distill-loss nll --alpha 0.5 --gps-iters 20 --episode-length 1000 --device auto --exp-name hopper_autoreset --num-conditions 2