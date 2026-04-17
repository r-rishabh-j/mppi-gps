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
MPPI runs on CPU; policy training uses the selected device.

```bash
# full run on acrobot, seeded from the BC dataset if present
python -m scripts.run_dagger --env acrobot \
    --dagger-iters 10 --rollouts-per-iter 20 --episode-len 200 \
    --seed-from data/acrobot_bc.h5 --device auto

# quick smoke test (~1–2 min)
python -m scripts.run_dagger --env acrobot \
    --dagger-iters 2 --rollouts-per-iter 3 --episode-len 80 \
    --distill-epochs 3 --n-eval-eps 3 --eval-ep-len 150 --device auto
```

Flags:
- `--device auto|cpu|mps|cuda` — auto resolves to cuda → mps → cpu.
- `--beta-schedule linear|constant_zero` — β decays 1→0 over K/2 iters, or always 0.
- `--buffer-cap 200000` — aggregated buffer capacity (oldest evicted).
- `--seed-from PATH` — warm-start buffer from an existing BC h5.

Outputs:
- `checkpoints/dagger/dagger_<env>_iter<k>.pt` — per-iteration policy weights.
- `results/dagger_acrobot/dagger_log.csv` — per-iter metrics + MPPI baseline.

## GPS (KL-constrained distillation)

```bash
python -m scripts.run_gps --env acrobot --device auto
python -m scripts.run_gps --env hopper --gps-iters 30 --device auto
```

### Policy-prior-only GPS (no KL / no BADMM)

Bias MPPI with `-α·log π(u|s)`, then plain behavior-clone the resulting
(state, action) pairs. No KL estimation, no dual variable updates — `ν`
stays constant at `--nu` (default 1.0).

```bash
python -m scripts.run_gps --env acrobot --disable-kl --distill-loss mse \
    --alpha 0.1 --nu 1.0 --device auto
```

Flags:
- `--disable-kl` — skip KL accumulation + BADMM; run policy-prior-only variant.
- `--distill-loss {nll,mse}` — `mse` matches plain BC, `nll` uses weighted NLL (default).
- `--alpha` — weight on `-log π` inside the MPPI cost (policy-prior strength).
- `--nu` — constant multiplier on the prior; effective weight is `alpha * nu`.
- `--device auto|cpu|mps|cuda` — policy training device (MPPI always runs on CPU).

## Baselines, tuning, ablations

```bash
python -m scripts.run_sb3_baseline --env Hopper-v5 --algo SAC

python -m scripts.tuning.tune_acrobot
python -m scripts.tuning.tune_cheetah
python -m scripts.tuning.tune_hopper

python -m scripts.run_ablations --env acrobot
python -m scripts.visualisation.plot_results --env acrobot
```
