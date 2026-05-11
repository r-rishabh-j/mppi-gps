"""Evaluate every iter_*.pt checkpoint in a run dir and plot eval cost vs.
true environment-physics steps consumed during training.

Walks a run directory like ``PAPER/hopper/gps-combined/`` and:

  1. Discovers ``iter_000.pt … iter_NNN.pt``.
  2. For each checkpoint, runs ``--n-eval`` policy episodes and records
     mean ± std cost (mean used for the curve, std for an error band).
  3. Runs the MPPI baseline ``--n-mppi-eval`` times and overlays the
     mean as a horizontal dotted line.
  4. Plots cost vs. cumulative env-physics steps.

Pass ``--bc-run-dir`` to overlay a behaviour-cloning run on the same
plot. Both runs are evaluated with identical (``--n-eval``, ``--eval-len``,
``--seed``) so the per-checkpoint comparison is apples-to-apples; each
run's x-axis is computed from its OWN ``config.json`` (since BC and GPS
typically use different ``num_conditions`` / open-loop budgets per iter).

X-axis steps formula (per training iteration):

    GPS:    steps_per_iter = num_conditions
                           * ceil(episode_length / open_loop_steps)
                           * K * H
    DAgger: steps_per_iter = rollouts_per_iter * episode_len * K * H
            (no /open_loop divisor — relabel forces a full rollout per
            env step.)

Checkpoint ``iter_k`` is plotted at ``k * steps_per_iter`` — so every
curve starts at x=0 regardless of per-iter cost (e.g. DAgger's 82M
per-iter on hopper vs GPS's 27M does not push DAgger right by 55M at
iter 0). DAgger's ``warmup_rollouts`` phase (a one-time fixed offset)
is also intentionally NOT added — it would visually push the whole
DAgger curve to the right and obscure per-iter cost differences. The
warmup count is still computed and printed in the per-run breakdown
for transparency.

CSV is intentionally NOT read — costs are recomputed live so this script
is the source of truth for the curve.

Caching:
    Each run dir gets an ``eval_cache.json`` keyed on every parameter that
    affects the result. Re-runs reuse cached values; only parameter changes
    (or new checkpoints) trigger fresh evaluation. So if you eval at
    ``--n-eval 5 --eval-len 1000 --seed 42``, then re-run with the same
    args plus ``--n-mppi-eval 5``, only the MPPI baseline is recomputed.
    Bump ``--n-eval`` and only the policy evals re-run. Add new
    ``iter_*.pt`` files and only the new ones get evaluated. Use
    ``--no-cache`` to bypass entirely or ``--clear-cache`` to wipe before
    running.

    Cache key for policy evals: (iter, n_eval, eval_len, seed).
    Cache key for MPPI baseline: (mppi_config_dict, n_mppi_eval, eval_len, seed).
    The MPPI key includes the full MPPIConfig (K, H, lam, noise_sigma,
    noise_cov, open_loop_steps, …) so changing any of those forces a
    fresh baseline.

Examples:
    # GPS only
    python -m scripts.eval_run --run-dir PAPER/hopper/gps-combined \
        --n-eval 5 --n-mppi-eval 3

    # GPS + BC overlay
    python -m scripts.eval_run --run-dir PAPER/hopper/gps-combined \
        --bc-run-dir PAPER/hopper/BC --n-eval 5 --n-mppi-eval 3
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.envs import make_env
from src.policy.gaussian_policy import GaussianPolicy
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import MPPIConfig, PolicyConfig
from src.utils.device import pick_device
from src.utils.evaluation import evaluate_policy, evaluate_mppi
from src.utils.experiment import load_checkpoint
from src.mppi.mppi import MPPI


_ITER_RE = re.compile(r"^iter_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Primary run dir (typically GPS) containing iter_*.pt + config.json.")
    p.add_argument("--bc-run-dir", default=None,
                   help="Optional BC run dir with the same layout. When passed, "
                        "evaluated under identical (n_eval, eval_len, seed) and "
                        "overlaid on the same plot.")
    p.add_argument("--dagger-run-dir", default=None,
                   help="Optional DAgger run dir. Same eval protocol as --bc-run-dir; "
                        "x-axis uses DAgger's per-iter formula "
                        "(rollouts_per_iter * episode_len * K * H, no /open_loop "
                        "since relabel is per step) plus the one-time warmup_rollouts "
                        "offset at iter 0.")
    p.add_argument("--label", default="GPS",
                   help="Label for --run-dir's curve (default: GPS).")
    p.add_argument("--bc-label", default="BC",
                   help="Label for --bc-run-dir's curve (default: BC).")
    p.add_argument("--dagger-label", default="DAgger",
                   help="Label for --dagger-run-dir's curve (default: DAgger).")
    p.add_argument("--dagger-x-scale", type=float, default=1.0,
                   help="Multiplicative factor applied to DAgger's cum_steps "
                        "BEFORE plotting / json dump. Default 1.0 = honest "
                        "(matches `_steps_per_iter`'s rollouts*ep_len*K*H). "
                        "Use e.g. 1/open_loop_steps to visually compress "
                        "DAgger onto GPS's budget when you know the per-step "
                        "relabel cost shouldn't be charged differently. "
                        "Recorded in the JSON dump as `dagger_x_scale`.")
    p.add_argument("--gps-x-scale", type=float, default=1.0,
                   help="Same idea for the primary (GPS) curve. Default 1.0.")
    p.add_argument("--bc-x-scale", type=float, default=1.0,
                   help="Same idea for the BC overlay curve. Default 1.0.")
    p.add_argument("--n-eval", type=int, default=5,
                   help="Policy episodes per checkpoint (applied to BOTH runs).")
    p.add_argument("--n-mppi-eval", type=int, default=3,
                   help="MPPI baseline episodes (set 0 to skip).")
    p.add_argument("--eval-len", type=int, default=None,
                   help="Steps per episode. Defaults to gps.eval_ep_len from "
                        "config.json of --run-dir, or 500 if missing.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--max-iter", type=int, default=None,
                   help="Only eval checkpoints with iter index <= this (applied to BOTH runs).")
    p.add_argument("--stride", type=int, default=1,
                   help="Eval every Nth checkpoint (1 = all). Saves time on long runs.")
    p.add_argument("--out", default=None,
                   help="Output plot path. Default: <run-dir>/eval_curve.png.")
    p.add_argument("--results-out", default=None,
                   help="JSON dump of per-iter cost stats. "
                        "Default: <run-dir>/eval_curve.json.")
    p.add_argument("--mppi-config", default=None,
                   help="Override MPPI config name for the baseline (e.g. "
                        "'hopper'). Default: env name from config.json.")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass eval_cache.json entirely — recompute all "
                        "policy and MPPI evaluations. Useful when checkpoints "
                        "have been silently overwritten.")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete eval_cache.json in each run dir before evaluating.")
    return p.parse_args()


# ---------- caching ----------------------------------------------------------
#
# Per-run-dir cache file: <run_dir>/eval_cache.json with shape:
#   {
#     "policy_evals": [
#       {"iter": int, "n_eval": int, "eval_len": int, "seed": int,
#        "mean_cost": float, "std_cost": float, "per_ep_costs": [...]}, ...
#     ],
#     "mppi_evals": [
#       {"mppi_config": {...full dict...}, "n_mppi_eval": int,
#        "eval_len": int, "seed": int,
#        "mean_cost": float, "std_cost": float, "per_ep_costs": [...]}, ...
#     ],
#   }
#
# Linear scan for lookup — fine for the O(few hundred) entries this
# accumulates in practice. A change to any key field causes a miss and the
# new entry is appended (old entry stays around as a "history" record).

CACHE_FILENAME = "eval_cache.json"


def _load_cache(run_dir: Path, *, no_cache: bool) -> dict:
    if no_cache:
        return {"policy_evals": [], "mppi_evals": []}
    path = run_dir / CACHE_FILENAME
    if not path.exists():
        return {"policy_evals": [], "mppi_evals": []}
    try:
        cache = json.loads(path.read_text())
    except json.JSONDecodeError:
        # Corrupt cache — start fresh rather than crashing the eval run.
        print(f"[cache] corrupt {path}; starting empty.")
        return {"policy_evals": [], "mppi_evals": []}
    cache.setdefault("policy_evals", [])
    cache.setdefault("mppi_evals", [])
    return cache


def _save_cache(run_dir: Path, cache: dict, *, no_cache: bool) -> None:
    if no_cache:
        return
    path = run_dir / CACHE_FILENAME
    path.write_text(json.dumps(cache, indent=2))


def _lookup_policy(cache: dict, iter_idx: int, n_eval: int,
                   eval_len: int, seed: int) -> dict | None:
    for entry in cache["policy_evals"]:
        if (entry["iter"] == iter_idx
                and entry["n_eval"] == n_eval
                and entry["eval_len"] == eval_len
                and entry["seed"] == seed):
            return entry
    return None


def _lookup_mppi(cache: dict, mppi_cfg_dict: dict, n_mppi_eval: int,
                 eval_len: int, seed: int) -> dict | None:
    for entry in cache["mppi_evals"]:
        if (entry["mppi_config"] == mppi_cfg_dict
                and entry["n_mppi_eval"] == n_mppi_eval
                and entry["eval_len"] == eval_len
                and entry["seed"] == seed):
            return entry
    return None


def _mppi_cfg_to_dict(cfg: MPPIConfig) -> dict:
    """Convert MPPIConfig → plain dict for cache-key comparison.

    `noise_cov` (a list of lists) is preserved verbatim. Field order is
    fixed by the dataclass, so dict equality is well-defined.
    """
    return {
        "K": cfg.K,
        "H": cfg.H,
        "lam": cfg.lam,
        "noise_sigma": cfg.noise_sigma,
        "adaptive_lam": cfg.adaptive_lam,
        "n_eff_threshold": cfg.n_eff_threshold,
        "open_loop_steps": cfg.open_loop_steps,
        "noise_cov": cfg.noise_cov,
    }


def _discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (iter_idx, path) pairs for every iter_NNN.pt in run_dir."""
    out: list[tuple[int, Path]] = []
    for p in run_dir.iterdir():
        m = _ITER_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _build_policy(env, env_name: str, blob: dict, run_cfg: dict | None,
                  device) -> torch.nn.Module:
    """Mirror eval_checkpoint's class-resolution logic.

    Backward-compat: if the checkpoint predates action normalization (no
    `_act_scale` / `_act_bias` buffers in its state_dict) but the current
    `USE_ACT_NORM=True` build registered those buffers, we disable
    `_has_act_norm` for this policy instance so `_to_phys` / `_to_norm`
    short-circuit and the network's outputs are treated as physical
    actions (which is what they were trained to produce). Without this,
    loading an old DAgger / pre-norm checkpoint would raise
    "Missing key(s) ... _act_scale, _act_bias".
    """
    policy_class = blob.get("policy_class") or (run_cfg or {}).get("policy_class")
    use_det = policy_class == "DeterministicPolicy"
    PolicyCls = DeterministicPolicy if use_det else GaussianPolicy
    policy_cfg = PolicyConfig.for_env(env_name)
    bounds = env.action_bounds
    policy = PolicyCls(env.obs_dim, env.action_dim, policy_cfg,
                       device=device, action_bounds=bounds)

    sd = blob["state_dict"]
    has_norm_in_ckpt = "_act_scale" in sd and "_act_bias" in sd
    if not has_norm_in_ckpt and getattr(policy, "_has_act_norm", False):
        # Old checkpoint: outputs are already in physical space.
        policy._has_act_norm = False
    # strict=False so the missing _act_scale/_act_bias don't raise; with
    # _has_act_norm flipped above, those buffers are never read anyway.
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    # Anything other than the act-norm buffers is a real schema mismatch.
    real_missing = [k for k in missing if k not in ("_act_scale", "_act_bias")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"state_dict load mismatch: missing={real_missing}, "
            f"unexpected={list(unexpected)}"
        )

    bad = sorted(k for k, v in policy.state_dict().items()
                 if not torch.isfinite(v).all())
    if bad:
        raise ValueError(
            f"checkpoint has non-finite weights in {len(bad)} tensors "
            f"(showing first 5: {bad[:5]}). Skip this run."
        )
    policy.eval()
    return policy


def _steps_per_iter(run_cfg: dict) -> tuple[int, int, dict]:
    """Compute (steps_per_iter, warmup_steps, breakdown_dict) from training config.

    Auto-detects GPS vs DAgger config shape:

    * GPS (`configs.gps` block):
        per_iter = num_conditions
                   * ceil(episode_length / open_loop_steps)
                   * K * H
        warmup   = 0

      MPPI does one K*H sample-and-rollout pass every `open_loop_steps`
      env steps (intermediate steps replay cached actions).

    * DAgger (`configs.dagger` block):
        per_iter = rollouts_per_iter * episode_len * K * H
        warmup   = warmup_rollouts * episode_len * K * H / open_loop_steps
                   (only the warmup MPPI uses open-loop caching;
                    DAgger relabels every visited state, so no divisor.)

      DAgger's relabel call forces a full rollout per env step — the
      open-loop cadence applies only to the warmup-phase pure-MPPI
      rollouts, not to the per-step relabel afterwards.

    Returns:
        steps_per_iter: physics steps consumed by ONE training iter (after warmup).
        warmup_steps:   one-time offset added to cumulative count at iter 0.
        breakdown:      dict for printing/saving.
    """
    mppi_cfg = run_cfg["configs"]["mppi"]
    K = int(mppi_cfg["K"])
    H = int(mppi_cfg["H"])
    open_loop = int(mppi_cfg.get("open_loop_steps", 1) or 1)

    if "gps" in run_cfg["configs"]:
        gps_cfg = run_cfg["configs"]["gps"]
        N = int(gps_cfg["num_conditions"])
        rollout_len = int(gps_cfg["episode_length"])
        plan_calls_per_episode = int(np.ceil(rollout_len / open_loop))
        steps_per_iter = N * plan_calls_per_episode * K * H
        warmup_steps = 0
        breakdown = dict(
            kind="gps",
            num_conditions=N,
            episode_length=rollout_len,
            K=K, H=H, open_loop_steps=open_loop,
            plan_calls_per_episode=plan_calls_per_episode,
            steps_per_iter=steps_per_iter,
            warmup_steps=warmup_steps,
        )
        return steps_per_iter, warmup_steps, breakdown

    if "dagger" in run_cfg["configs"]:
        dag_cfg = run_cfg["configs"]["dagger"]
        N = int(dag_cfg["rollouts_per_iter"])
        rollout_len = int(dag_cfg["episode_len"])
        steps_per_iter = N * rollout_len * K * H
        warmup_rollouts = int(
            run_cfg.get("cli_args", {}).get("warmup_rollouts", 0) or 0
        )
        warmup_plan_calls = int(np.ceil(rollout_len / open_loop))
        warmup_steps = warmup_rollouts * warmup_plan_calls * K * H
        breakdown = dict(
            kind="dagger",
            rollouts_per_iter=N,
            episode_len=rollout_len,
            K=K, H=H, open_loop_steps=open_loop,
            warmup_rollouts=warmup_rollouts,
            steps_per_iter=steps_per_iter,
            warmup_steps=warmup_steps,
        )
        return steps_per_iter, warmup_steps, breakdown

    raise KeyError(
        "config.json has neither configs.gps nor configs.dagger — "
        "can't compute per-iter env-step cost."
    )


def _eval_one_run(
    run_dir: Path,
    env,
    *,
    n_eval: int,
    eval_len: int,
    seed: int,
    device,
    max_iter: int | None,
    stride: int,
    label: str,
    no_cache: bool,
) -> dict[str, Any]:
    """Evaluate every iter_*.pt in a run dir and return per-iter cost stats.

    Same eval protocol as evaluate_policy: deterministic per-episode seed
    (`seed + ep`), greedy policy actions, full episode_len rollouts. The
    same `env` instance is reused for both runs in the multi-curve case
    so cost comparisons aren't biased by env construction order.

    Cache: per-(iter, n_eval, eval_len, seed) hits in <run_dir>/eval_cache.json
    skip the policy build + episode rollouts. Misses recompute and append
    to the cache; cache is flushed to disk after every miss so a Ctrl-C
    halfway through doesn't lose work.
    """
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} missing — need it for env name + MPPI/GPS hyperparams."
        )
    run_cfg = json.loads(cfg_path.read_text())
    steps_per_iter, warmup_steps, breakdown = _steps_per_iter(run_cfg)

    ckpts = _discover_checkpoints(run_dir)
    if max_iter is not None:
        ckpts = [(i, p) for i, p in ckpts if i <= max_iter]
    if stride > 1:
        ckpts = ckpts[::stride]
    if not ckpts:
        raise FileNotFoundError(f"No iter_*.pt found in {run_dir}")

    print(f"[{label}] dir = {run_dir}")
    print(f"[{label}] steps/iter = {steps_per_iter:,}   breakdown = {breakdown}")
    print(f"[{label}] found {len(ckpts)} checkpoints (iter "
          f"{ckpts[0][0]}..{ckpts[-1][0]})")

    cache = _load_cache(run_dir, no_cache=no_cache)
    n_hits = 0
    n_misses = 0

    iter_indices: list[int] = []
    cum_steps: list[int] = []
    mean_costs: list[float] = []
    std_costs: list[float] = []
    per_ep_costs: list[list[float]] = []

    env_name = run_cfg["env"]
    for it_idx, ckpt_path in ckpts:
        cached = _lookup_policy(cache, it_idx, n_eval, eval_len, seed)
        if cached is not None:
            mean_cost = cached["mean_cost"]
            std_cost = cached["std_cost"]
            per_ep = cached["per_ep_costs"]
            n_hits += 1
            tag = "cache "
            dt = 0.0
        else:
            t0 = time.time()
            blob = load_checkpoint(ckpt_path, map_location=device)
            policy = _build_policy(env, env_name, blob, run_cfg, device)
            stats = evaluate_policy(
                policy, env,
                n_episodes=n_eval,
                episode_len=eval_len,
                seed=seed,
                render=False,
            )
            dt = time.time() - t0
            mean_cost = stats["mean_cost"]
            std_cost = stats["std_cost"]
            per_ep = stats["per_ep"]
            cache["policy_evals"].append({
                "iter": it_idx,
                "n_eval": n_eval,
                "eval_len": eval_len,
                "seed": seed,
                "mean_cost": mean_cost,
                "std_cost": std_cost,
                "per_ep_costs": per_ep,
            })
            # Flush after every miss so a Ctrl-C doesn't waste work.
            _save_cache(run_dir, cache, no_cache=no_cache)
            n_misses += 1
            tag = "fresh "

        # Plot iter k at `k * steps_per_iter` so every curve starts at x=0
        # (iter 0 = "cost before any training steps were charged"). Previously
        # this was `(k+1) * steps_per_iter`, which forced each curve's first
        # point to its own per-iter cost and made methods with different
        # per-iter budgets (e.g. DAgger 82M vs GPS 27M on hopper) look
        # offset at the start. Warmup is still NOT added — see header.
        steps = it_idx * steps_per_iter
        iter_indices.append(it_idx)
        cum_steps.append(steps)
        mean_costs.append(mean_cost)
        std_costs.append(std_cost)
        per_ep_costs.append(per_ep)
        print(f"  [{label}] {tag} iter {it_idx:3d} | steps={steps:>12,} | "
              f"cost = {mean_cost:>9.2f} ± {std_cost:>7.2f}  "
              f"({dt:.1f}s)" + (" [cached]" if tag == "cache " else ""))

    print(f"[{label}] cache: {n_hits} hits, {n_misses} misses")

    return {
        "label": label,
        "run_dir": str(run_dir),
        "env": env_name,
        "iters": iter_indices,
        "cum_steps": cum_steps,
        "mean_costs": mean_costs,
        "std_costs": std_costs,
        "per_ep_costs": per_ep_costs,
        "steps_per_iter": steps_per_iter,
        "steps_breakdown": breakdown,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    # Read primary config to fix env + default eval_len.
    cfg_path = run_dir / "config.json"
    run_cfg = json.loads(cfg_path.read_text())
    env_name = run_cfg["env"]
    # eval_ep_len lives under either configs.gps or configs.dagger depending
    # on the trainer that produced this dir; fall back to 500 if neither.
    cfg_block = (
        run_cfg.get("configs", {}).get("gps")
        or run_cfg.get("configs", {}).get("dagger")
        or {}
    )
    eval_len = args.eval_len or int(cfg_block.get("eval_ep_len", 500))

    # Cross-check overlay dirs use the same env (otherwise the same `env`
    # instance and the same cost-axis comparison make no sense).
    def _resolve_overlay(arg: str | None, kind: str) -> Path | None:
        if arg is None:
            return None
        d = Path(arg).resolve()
        if not d.is_dir():
            raise NotADirectoryError(d)
        cfg = json.loads((d / "config.json").read_text())
        if cfg["env"] != env_name:
            raise ValueError(
                f"--{kind}-run-dir env={cfg['env']!r} doesn't match "
                f"--run-dir env={env_name!r}; can't compare on the same plot."
            )
        return d

    bc_run_dir = _resolve_overlay(args.bc_run_dir, "bc")
    dagger_run_dir = _resolve_overlay(args.dagger_run_dir, "dagger")

    print(f"[eval-run] env = {env_name}   eval_len = {eval_len}   "
          f"n_eval = {args.n_eval}   seed = {args.seed}   "
          f"cache = {'OFF' if args.no_cache else 'ON'}")

    # Optional cache wipe — done up-front so subsequent loads start clean.
    if args.clear_cache:
        wipe_dirs = [run_dir]
        if bc_run_dir is not None:
            wipe_dirs.append(bc_run_dir)
        if dagger_run_dir is not None:
            wipe_dirs.append(dagger_run_dir)
        for d in wipe_dirs:
            cache_path = d / CACHE_FILENAME
            if cache_path.exists():
                cache_path.unlink()
                print(f"[cache] cleared {cache_path}")

    device = pick_device(args.device)
    env = make_env(env_name)

    # Primary run.
    primary = _eval_one_run(
        run_dir, env,
        n_eval=args.n_eval, eval_len=eval_len, seed=args.seed,
        device=device, max_iter=args.max_iter, stride=args.stride,
        label=args.label, no_cache=args.no_cache,
    )

    # Optional BC overlay.
    secondary: dict[str, Any] | None = None
    if bc_run_dir is not None:
        secondary = _eval_one_run(
            bc_run_dir, env,
            n_eval=args.n_eval, eval_len=eval_len, seed=args.seed,
            device=device, max_iter=args.max_iter, stride=args.stride,
            label=args.bc_label, no_cache=args.no_cache,
        )

    # Optional DAgger overlay.
    tertiary: dict[str, Any] | None = None
    if dagger_run_dir is not None:
        tertiary = _eval_one_run(
            dagger_run_dir, env,
            n_eval=args.n_eval, eval_len=eval_len, seed=args.seed,
            device=device, max_iter=args.max_iter, stride=args.stride,
            label=args.dagger_label, no_cache=args.no_cache,
        )

    # MPPI baseline (no policy prior). Cached in the PRIMARY run dir keyed
    # on the full MPPIConfig dict — so changing K, H, lam, noise_sigma,
    # noise_cov, open_loop_steps, etc. forces a fresh baseline.
    mppi_mean: float | None = None
    mppi_std: float | None = None
    mppi_per_ep: list[float] = []
    if args.n_mppi_eval > 0:
        try:
            mppi_cfg = MPPIConfig.load(args.mppi_config or env_name)
        except FileNotFoundError:
            print(f"[eval-run] no configs/{env_name}_best.json — falling back to "
                  f"in-config MPPI hyperparams.")
            mppi_cfg = MPPIConfig(**run_cfg["configs"]["mppi"])
        mppi_cfg_dict = _mppi_cfg_to_dict(mppi_cfg)

        primary_cache = _load_cache(run_dir, no_cache=args.no_cache)
        cached_mppi = _lookup_mppi(primary_cache, mppi_cfg_dict,
                                   args.n_mppi_eval, eval_len, args.seed)
        if cached_mppi is not None:
            mppi_mean = cached_mppi["mean_cost"]
            mppi_std = cached_mppi["std_cost"]
            mppi_per_ep = cached_mppi["per_ep_costs"]
            print(f"[eval-run] MPPI baseline (cached): cost = {mppi_mean:.2f} "
                  f"± {mppi_std:.2f}")
        else:
            controller = MPPI(env, mppi_cfg)
            print(f"[eval-run] running MPPI baseline ({args.n_mppi_eval} eps, "
                  f"K={mppi_cfg.K}, H={mppi_cfg.H}, open_loop={mppi_cfg.open_loop_steps})…")
            t0 = time.time()
            bstats = evaluate_mppi(env, controller, args.n_mppi_eval,
                                   episode_len=eval_len, seed=args.seed)
            mppi_mean = bstats["mean_cost"]
            mppi_std = bstats["std_cost"]
            mppi_per_ep = bstats["per_ep"]
            print(f"  MPPI baseline: cost = {mppi_mean:.2f} ± {mppi_std:.2f}  "
                  f"({time.time() - t0:.1f}s)")
            primary_cache["mppi_evals"].append({
                "mppi_config": mppi_cfg_dict,
                "n_mppi_eval": args.n_mppi_eval,
                "eval_len": eval_len,
                "seed": args.seed,
                "mean_cost": mppi_mean,
                "std_cost": mppi_std,
                "per_ep_costs": mppi_per_ep,
            })
            _save_cache(run_dir, primary_cache, no_cache=args.no_cache)

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot_curve(res: dict, color: str, marker: str, x_scale: float) -> None:
        cum = np.array(res["cum_steps"], dtype=float) * x_scale
        mu = np.array(res["mean_costs"])
        sd = np.array(res["std_costs"])
        # suffix = f"  (×{x_scale:g})" if x_scale != 1.0 else ""
        suffix = f""
        ax.plot(cum, mu, f"-{marker}", color=color, label=f"{res['label']}")
                # label=f"{res['label']} (n_eval={args.n_eval}){suffix}")
        ax.fill_between(cum, mu - sd, mu + sd, color=color, alpha=0.20)

    _plot_curve(primary, color="C0", marker="o", x_scale=args.gps_x_scale)
    if secondary is not None:
        _plot_curve(secondary, color="C2", marker="s", x_scale=args.bc_x_scale)
    if tertiary is not None:
        _plot_curve(tertiary, color="C1", marker="^", x_scale=args.dagger_x_scale)

    if mppi_mean is not None:
        ax.axhline(mppi_mean, color="C3", linestyle="--", linewidth=1.5,
                   label=f"MPPI baseline")
                        #  f"mean={mppi_mean:.1f})")

    # Right-align curves: clip x-axis to the SHORTEST run's right edge so
    # every curve terminates at the same x. Avoids the longest-run trailing
    # off solo to the right and visually swamping the others. Each curve's
    # data is unchanged — matplotlib just doesn't render past x_max.
    curve_endpoints: list[float] = []
    for res, scale in [
        (primary, args.gps_x_scale),
        (secondary, args.bc_x_scale),
        (tertiary, args.dagger_x_scale),
    ]:
        if res is not None and res["cum_steps"]:
            curve_endpoints.append(max(res["cum_steps"]) * scale)
    if curve_endpoints:
        ax.set_xlim(left=0, right=min(curve_endpoints))

    ax.set_xlabel("Cumulative Environment Steps (1e8)", fontsize=14)
    ax.set_ylabel("Episode Cost", fontsize=14)
    title = f"{env_name} - MPPI-GPS"
    if secondary is not None:
        title += f"  vs  {bc_run_dir.name}"
    if tertiary is not None:
        title += f"  vs  {dagger_run_dir.name}"
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    # Park the legend outside the axes (upper-right of the figure) so it
    # never overlaps the curves regardless of where they end up.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, frameon=True)
    fig.tight_layout()

    out_plot = Path(args.out) if args.out else (run_dir / "eval_curve.png")
    # bbox_inches='tight' makes savefig include the out-of-axes legend in
    # the rendered image (otherwise it gets clipped).
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval-run] plot → {out_plot}")

    # ----- json dump (so we can replot without re-running) -----
    out_json = Path(args.results_out) if args.results_out \
        else (run_dir / "eval_curve.json")
    payload: dict[str, Any] = {
        "env": env_name,
        "n_eval": args.n_eval,
        "n_mppi_eval": args.n_mppi_eval,
        "eval_len": eval_len,
        "seed": args.seed,
        "primary": primary,
        "secondary": secondary,
        "tertiary": tertiary,
        "mppi_mean": mppi_mean,
        "mppi_std": mppi_std,
        "mppi_per_ep": mppi_per_ep,
        # Provenance — so a future replot knows the x-axis was scaled.
        # Honest cum_steps stay inside primary/secondary/tertiary; the scales
        # below are the multipliers that were applied at plot time.
        "gps_x_scale": args.gps_x_scale,
        "bc_x_scale": args.bc_x_scale,
        "dagger_x_scale": args.dagger_x_scale,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[eval-run] data → {out_json}")

    env.close()


if __name__ == "__main__":
    main()
