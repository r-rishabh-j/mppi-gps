"""Collect (obs, expert action) trajectories from MPPI for pure SL/BC.

Writes an h5 dataset compatible with `test_sl.py` and DAgger's `--seed-from`:
    keys:   states (M, T, obs_dim), actions (M, T, act_dim), costs (M, T)
    attrs:  env, M, T, obs_dim, act_dim, mppi_cfg (json)

Caching:
    If the output path already exists, the script prints its stats and exits
    without re-collecting. Pass `--force` to overwrite. The default output path
    is `data/<env>_bc.h5`, so collecting for a new env does not trample prior
    datasets.

Examples:
    python -m scripts.collect_bc_demos --env acrobot
    python -m scripts.collect_bc_demos --env hopper --auto-reset -M 30 -T 500
    python -m scripts.collect_bc_demos --env acrobot --force         # re-collect
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


_ENVS = ["acrobot", "half_cheetah", "point_mass", "hopper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=_ENVS)
    p.add_argument("-M", "--num-trajectories", type=int, default=50,
                   dest="M", help="number of MPPI rollouts to collect")
    p.add_argument("-T", "--trajectory-length", type=int, default=1000,
                   dest="T", help="max steps per rollout")
    p.add_argument("--out", default=None,
                   help="output h5 path (default: data/<env>_bc.h5)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing dataset at --out")
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
    print("  pass --force to re-collect.")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path(f"data/{args.env}_bc.h5")

    if out_path.exists() and not args.force:
        _describe_cached(out_path)
        return

    np.random.seed(args.seed)

    env = make_env(args.env)
    mppi_cfg = MPPIConfig.load(args.env)
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

    print(f"collecting {M} rollouts x {T} steps on {args.env}  "
          f"(obs_dim={obs_dim}, act_dim={act_dim}, auto_reset={args.auto_reset})")
    print(f"→ {out_path}")

    outer = tqdm(range(M), desc="rollouts", unit="traj")
    for i in outer:
        # Deterministic per-trajectory init so reruns are reproducible.
        np.random.seed(args.seed + i)
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("states", data=states)
        f.create_dataset("actions", data=actions)
        f.create_dataset("costs", data=costs)
        f.attrs["env"] = args.env
        f.attrs["M"] = M
        f.attrs["T"] = T
        f.attrs["obs_dim"] = obs_dim
        f.attrs["act_dim"] = act_dim
        f.attrs["seed"] = args.seed
        f.attrs["auto_reset"] = args.auto_reset
        f.attrs["mppi_cfg"] = json.dumps(asdict(mppi_cfg))

    print(f"\nsaved {M} trajectories of length {T} to {out_path}")
    print(f"mean total cost across trajectories: {costs.sum(axis=1).mean():.2f}")
    env.close()


if __name__ == "__main__":
    main()
