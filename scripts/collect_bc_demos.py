"""Collect (obs, expert action) trajectories from MPPI for pure SL/BC.

Writes an h5 dataset compatible with `test_sl.py` and DAgger's `--seed-from`:
    keys:   states (M, T, obs_dim), actions (M, T, act_dim), costs (M, T)
    attrs:  env, M, T, obs_dim, act_dim, mppi_cfg (json)

Caching:
    If the output path already exists, the script prints its stats and exits
    without re-collecting. Pass `--force` to overwrite, or `--append` to add
    M more trajectories to the existing dataset (seeds are auto-offset by the
    existing trajectory count so new rollouts don't duplicate old ones). The
    default output path is `data/<env>_bc.h5`, so collecting for a new env
    does not trample prior datasets.

Examples:
    python -m scripts.collect_bc_demos --env acrobot
    python -m scripts.collect_bc_demos --env hopper --auto-reset -M 30 -T 500
    python -m scripts.collect_bc_demos --env acrobot --force              # re-collect
    python -m scripts.collect_bc_demos --env acrobot --append -M 20       # add 20 more
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from src.envs import make_env
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


_ENVS = ["acrobot", "adroit_pen", "adroit_relocate", "half_cheetah", "point_mass", "hopper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=_ENVS)
    p.add_argument("-M", "--num-trajectories", type=int, default=50,
                   dest="M", help="number of MPPI rollouts to collect")
    p.add_argument("-T", "--trajectory-length", type=int, default=1000,
                   dest="T", help="max steps per rollout")
    p.add_argument("--out", default=None,
                   help="output h5 path (default: data/<env>_bc.h5)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true",
                      help="overwrite an existing dataset at --out")
    mode.add_argument("--append", action="store_true",
                      help="append -M new trajectories to an existing dataset at --out "
                           "(seeds auto-offset by the cached M so new rollouts are fresh). "
                           "env / T / obs_dim / act_dim / auto_reset must match; "
                           "mppi_cfg mismatch is a warning, not an error.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto-reset", action="store_true",
                   help="on env termination during a rollout, reset env+MPPI and keep "
                        "collecting until T steps are taken. Recommended for hopper.")
    return p.parse_args()


def _describe_cached(path: Path) -> None:
    with h5py.File(path, "r") as f:
        M = f.attrs.get("M", f["states"].shape[0])
        T = f.attrs.get("T", f["states"].shape[1])
        env_name = f.attrs.get("env", "?")
        mean_cost = float(f["costs"][:].sum(axis=1).mean()) if "costs" in f else float("nan")
    print(f"cache hit: {path}")
    print(f"  env={env_name}  M={M}  T={T}  mean_total_cost={mean_cost:.2f}")
    print("  pass --force to re-collect, or --append -M N to add N more.")


def _load_existing(path: Path) -> dict:
    """Read a full BC dataset into memory for appending."""
    with h5py.File(path, "r") as f:
        return {
            "states": f["states"][:],
            "actions": f["actions"][:],
            "costs": f["costs"][:],
            "attrs": {k: f.attrs[k] for k in f.attrs.keys()},
        }


def _validate_append_compat(existing_attrs: dict, args, env, mppi_cfg) -> None:
    """Raise if `existing_attrs` is incompatible for append; warn on soft mismatches.

    Hard-reject on any dimension/semantics change that would corrupt downstream
    consumers (env switch, T change, auto_reset change, obs/act dim change).
    mppi_cfg differences are soft — the user may intentionally be collecting
    with a retuned controller, and the dataset format is agnostic to it.
    """
    ex_env = existing_attrs.get("env", "?")
    if isinstance(ex_env, bytes):
        ex_env = ex_env.decode()

    mismatches = []
    if ex_env != args.env:
        mismatches.append(f"env: cached={ex_env!r} vs current={args.env!r}")
    ex_T = int(existing_attrs.get("T", 0))
    if ex_T != args.T:
        mismatches.append(f"T: cached={ex_T} vs current={args.T}")
    ex_obs = int(existing_attrs.get("obs_dim", env.obs_dim))
    if ex_obs != env.obs_dim:
        mismatches.append(f"obs_dim: cached={ex_obs} vs current={env.obs_dim}")
    ex_act = int(existing_attrs.get("act_dim", env.action_dim))
    if ex_act != env.action_dim:
        mismatches.append(f"act_dim: cached={ex_act} vs current={env.action_dim}")
    ex_ar = bool(existing_attrs.get("auto_reset", False))
    if ex_ar != args.auto_reset:
        mismatches.append(
            f"auto_reset: cached={ex_ar} vs current={args.auto_reset}")

    if mismatches:
        raise ValueError(
            "cannot --append: incompatible dataset attrs:\n  - "
            + "\n  - ".join(mismatches)
            + "\nuse --force to overwrite instead, or change your flags to match."
        )

    ex_cfg_raw = existing_attrs.get("mppi_cfg")
    if ex_cfg_raw is not None:
        if isinstance(ex_cfg_raw, bytes):
            ex_cfg_raw = ex_cfg_raw.decode()
        cur_cfg_norm = json.dumps(asdict(mppi_cfg), sort_keys=True)
        ex_cfg_norm = json.dumps(json.loads(ex_cfg_raw), sort_keys=True)
        if ex_cfg_norm != cur_cfg_norm:
            print("[append] WARNING: cached mppi_cfg differs from current config; "
                  "appended rollouts will be collected with the CURRENT config.")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path(f"data/{args.env}_bc.h5")

    # Resolve the three-way gate on an existing file: describe (default),
    # overwrite (--force), or append (--append). If the file doesn't exist,
    # --append silently falls through to a normal write so users can script
    # "collect or grow to N" workflows without pre-checking existence.
    existing: dict | None = None
    if out_path.exists():
        if args.force:
            pass  # overwrite
        elif args.append:
            existing = _load_existing(out_path)
        else:
            _describe_cached(out_path)
            return
    elif args.append:
        print(f"[append] {out_path} does not exist — creating a fresh dataset.")

    np.random.seed(args.seed)

    env = make_env(args.env)
    mppi_cfg = MPPIConfig.load(args.env)

    # Validate append compatibility now that env + mppi_cfg are constructed.
    existing_M = 0
    if existing is not None:
        _validate_append_compat(existing["attrs"], args, env, mppi_cfg)
        existing_M = int(existing["attrs"].get("M", existing["states"].shape[0]))

    controller = MPPI(env, cfg=mppi_cfg)

    obs_dim = env.obs_dim
    act_dim = env.action_dim
    M, T = args.M, args.T

    # Preallocate so the file shape is fixed and the in-RAM footprint is bounded.
    # When auto_reset fires, early-terminated segments are stitched contiguously,
    # so the (M, T) layout still holds exactly T steps per row.
    states = np.zeros((M, T, obs_dim), dtype=np.float32)
    actions = np.zeros((M, T, act_dim), dtype=np.float32)
    costs = np.zeros((M, T), dtype=np.float32)

    if existing is not None:
        print(f"appending {M} rollouts to existing {existing_M} on {args.env}  "
              f"(obs_dim={obs_dim}, act_dim={act_dim}, auto_reset={args.auto_reset})")
    else:
        print(f"collecting {M} rollouts x {T} steps on {args.env}  "
              f"(obs_dim={obs_dim}, act_dim={act_dim}, auto_reset={args.auto_reset})")
    print(f"→ {out_path}")

    outer = tqdm(range(M), desc="rollouts", unit="traj")
    for i in outer:
        # Deterministic per-trajectory init so reruns are reproducible.
        # Offset by existing_M when appending so new rollouts don't duplicate
        # the seeds that produced the cached trajectories.
        np.random.seed(args.seed + existing_M + i)
        env.reset()
        controller.reset()

        for t in range(T):
            obs = env._get_obs()
            state = env.get_state()
            action, info = controller.plan_step(state)
            _, cost, done, _ = env.step(action)

            states[i, t] = obs
            actions[i, t] = action
            costs[i, t] = cost

            if done and t < T - 1:
                if args.auto_reset:
                    env.reset()
                    controller.reset()
                    continue
                # no auto-reset: fill the remainder with the last (obs, 0, 0).
                # Downstream BC drops trailing zeros via cost/action sanity, but
                # for the common non-terminating envs we just never hit this.
                break

        total_cost = float(costs[i].sum())
        outer.set_postfix(total_cost=f"{total_cost:.1f}",
                          last_cost_min=f"{info['cost_min']:.2f}")

    # Concatenate with the existing dataset before writing. We overwrite the
    # file in "w" mode either way; there's no resize because legacy datasets
    # weren't created with maxshape=None (read+concat+rewrite keeps those
    # files compatible).
    if existing is not None:
        states_out = np.concatenate([existing["states"], states], axis=0)
        actions_out = np.concatenate([existing["actions"], actions], axis=0)
        costs_out = np.concatenate([existing["costs"], costs], axis=0)
        total_M = existing_M + M
    else:
        states_out, actions_out, costs_out = states, actions, costs
        total_M = M

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("states", data=states_out)
        f.create_dataset("actions", data=actions_out)
        f.create_dataset("costs", data=costs_out)
        f.attrs["env"] = args.env
        f.attrs["M"] = total_M
        f.attrs["T"] = T
        f.attrs["obs_dim"] = obs_dim
        f.attrs["act_dim"] = act_dim
        f.attrs["seed"] = args.seed
        f.attrs["auto_reset"] = args.auto_reset
        f.attrs["mppi_cfg"] = json.dumps(asdict(mppi_cfg))

    if existing is not None:
        new_cost_mean = float(costs.sum(axis=1).mean())
        total_cost_mean = float(costs_out.sum(axis=1).mean())
        print(f"\nappended {M} trajectories (new mean cost: {new_cost_mean:.2f}); "
              f"dataset now holds {total_M} trajectories of length {T} at {out_path}")
        print(f"mean total cost across ALL trajectories: {total_cost_mean:.2f}")
    else:
        print(f"\nsaved {M} trajectories of length {T} to {out_path}")
        print(f"mean total cost across trajectories: {costs.sum(axis=1).mean():.2f}")
    env.close()


if __name__ == "__main__":
    main()
