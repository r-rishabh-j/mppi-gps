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
python -m scripts.run_mppi           # generic entry
```

## Behavior cloning (one-shot)

```bash
# 1. collect (state, action) demos with MPPI → data/acrobot_bc.h5
python -m scripts.collect_bc_demos

# 2. train BC policy with MSE on policy mean
python -m scripts.test_sl
```

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

# quick smoke test (~1–2 min)
python -m scripts.run_dagger --env acrobot \
    --dagger-iters 2 --rollouts-per-iter 3 --episode-len 80 \
    --distill-epochs 3 --n-eval-eps 3 --eval-ep-len 150 --device auto
```

Flags:
- `--device auto|cpu|mps|cuda` — auto resolves to cuda → mps → cpu.
- `--deterministic` — use `DeterministicPolicy` (single-head, direct action regression) instead of `GaussianPolicy`.
- `--beta-schedule linear|constant_zero` — β decays 1→0 over K/2 iters, or always 0.
- `--buffer-cap 200000` — aggregated buffer capacity (oldest evicted).
- `--init-ckpt PATH` — load policy weights before warmup / DAgger. The checkpoint class must match `--deterministic` (deterministic vs Gaussian head shapes differ).
- `--seed-from PATH` — warm-start buffer from an existing BC h5 (no pre-training, just seeded rows).
- `--warmup-rollouts N` — collect N pure-MPPI rollouts **and BC-train the policy on them** before the DAgger loop. Rows land in the aggregate buffer with `round_idx=-1`.
- `--warmup-epochs E` — epochs of BC pre-training on warmup rollouts (default 20; ignored when `--warmup-rollouts=0`).

Outputs:
- `checkpoints/dagger/dagger_<env>_iter<k>.pt` — per-iteration policy weights.
- `results/dagger_acrobot/dagger_log.csv` — per-iter metrics + MPPI baseline.

## GPS (KL-constrained distillation)

```bash
python -m scripts.run_gps --env acrobot --device auto
python -m scripts.run_gps --env hopper --gps-iters 30 --device auto
```

Outputs (in `--ckpt-dir`, default `checkpoints/`):
- `gps_<env>_iter<k:03d>.pt` — per-iteration policy weights.
- `gps_<env>_best.pt` — weights from the iteration with the lowest mean rollout cost.
- `gps_<env>.pt` — final iteration weights.
- `gps_<env>_curves.json` / `.png` — cost / KL / nu / distill-loss curves.
- `gps_<env>.mp4` — first eval-episode video of the trained policy.

### Policy-prior-only GPS (no KL / no BADMM)

Bias MPPI with `-α·log π(u|s)`, then plain behavior-clone the resulting
(state, action) pairs. No KL estimation, no dual variable updates — `ν`
stays constant at `--nu` (default 1.0).

```bash
python -m scripts.run_gps --env acrobot --disable-kl --distill-loss nll \
    --alpha 0.1 --device auto --gps-iters 20

# quick smoke test
python -m scripts.run_gps --env acrobot --disable-kl --distill-loss nll \
    --alpha 0.1 --nu 1.0 --gps-iters 3 --num-conditions 2 --episode-length 80 \
    --n-eval 2 --eval-len 80 --device cpu
```

Flags:
- `--disable-kl` — skip KL accumulation + BADMM; run policy-prior-only variant.
- `--distill-loss {nll,mse}` — `mse` matches plain BC, `nll` uses weighted NLL (default).
- `--alpha` — weight on `-log π` inside the MPPI cost (policy-prior strength).
- `--nu` — constant multiplier on the prior; effective weight is `alpha * nu`.
- `--gps-iters`, `--num-conditions`, `--episode-length` — override `GPSConfig` defaults.
- `--n-eval`, `--eval-len` — evaluation episode count / length.
- `--ckpt-dir` — where per-iter / best / final checkpoints are saved (default `checkpoints/`).
- `--device auto|cpu|mps|cuda` — policy training device (MPPI always runs on CPU).

## Evaluate a saved checkpoint

Load any `.pt` policy state_dict and evaluate it (observation-normalizer
stats and tanh-squash buffers are restored automatically via `state_dict`).

```bash
# stochastic policy (GaussianPolicy — GPS default and DAgger default)
python -m scripts.eval_checkpoint --env acrobot \
    --ckpt checkpoints/gps_acrobot_best.pt \
    --n-eval 10 --render

# deterministic policy checkpoint (DAgger with --deterministic)
python -m scripts.eval_checkpoint --env acrobot --deterministic \
    --ckpt checkpoints/dagger/dagger_acrobot_iter09.pt \
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
