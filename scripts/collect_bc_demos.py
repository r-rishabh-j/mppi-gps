"""Collect (obs, expert action) trajectories from MPPI for pure SL/BC.

Writes an h5 dataset compatible with `test_sl.py` and DAgger's `--seed-from`:
    keys:   states     (M, T, obs_dim)
            actions    (M, T, act_dim)
            costs      (M, T)
            sensordata (M, T, nsensor)   — only when env produces non-empty
                                          sensordata (MuJoCo envs); absent
                                          for gym-wrapped envs and others
                                          where ``env.data.sensordata`` is
                                          missing or zero-sized.
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
    python -m scripts.collect_bc_demos --env hopper --warp --auto-reset -M 30 -T 500
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


_ENVS = ["acrobot", "adroit_pen", "adroit_relocate", "point_mass", "hopper", "ur5_push"]


def _try_capture_sensordata(env) -> np.ndarray | None:
    """Snapshot ``env.data.sensordata`` if the env exposes a non-empty one.

    Returns ``None`` for envs without MuJoCo-style sensor outputs (e.g.
    ``gym_wrapper``) or whose sensor list is empty. Used both as a probe
    at startup (to decide whether to allocate the per-step buffer) and
    inside the rollout loop (per-step capture).
    """
    data = getattr(env, "data", None)
    if data is None:
        return None
    sd = getattr(data, "sensordata", None)
    if sd is None or sd.size == 0:
        return None
    return sd.copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot", choices=_ENVS)
    p.add_argument("-M", "--num-trajectories", type=int, default=50,
                   dest="M", help="number of MPPI rollouts to collect")
    p.add_argument("-T", "--trajectory-length", type=int, default=1000,
                   dest="T", help="max steps per rollout")
    p.add_argument("--out", default=None,
                   help="output h5 path (default: data/<env>_bc.h5)")
    p.add_argument("--warp", action="store_true",
                   help="Use mujoco_warp GPU batch rollout for MPPI's K-sample "
                        "evaluation. Requires `uv pip install warp-lang mujoco-warp` "
                        "and an NVIDIA GPU with CUDA. Only hopper and adroit_relocate "
                        "have Warp envs; others error out. nworld is pinned to "
                        "MPPIConfig.K — single-condition collection (this script "
                        "does sequential rollouts), no batched MPPI needed.")
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
        sd_info = (
            f"  sensordata: {f['sensordata'].shape}" if "sensordata" in f else ""
        )
    print(f"cache hit: {path}")
    print(f"  env={env_name}  M={M}  T={T}  mean_total_cost={mean_cost:.2f}")
    if sd_info:
        print(sd_info)
    print("  pass --force to re-collect, or --append -M N to add N more.")


def _load_existing(path: Path) -> dict:
    """Read a full BC dataset into memory for appending."""
    with h5py.File(path, "r") as f:
        out: dict = {
            "states": f["states"][:],
            "actions": f["actions"][:],
            "costs": f["costs"][:],
            "attrs": {k: f.attrs[k] for k in f.attrs.keys()},
        }
        # Optional — only present in datasets collected after sensordata
        # capture was added. Absent for legacy files; appending against a
        # legacy file forces sensordata to be dropped (see validator).
        if "sensordata" in f:
            out["sensordata"] = f["sensordata"][:]
        return out


def _validate_append_compat(
    existing: dict, args, env, mppi_cfg, new_nsensor: int
) -> None:
    """Raise if `existing` is incompatible for append; warn on soft mismatches.

    Hard-reject on any dimension/semantics change that would corrupt downstream
    consumers (env switch, T change, auto_reset change, obs/act dim change,
    sensordata presence/dim change). mppi_cfg differences are soft — the user
    may intentionally be collecting with a retuned controller, and the dataset
    format is agnostic to it.
    """
    existing_attrs = existing["attrs"]
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

    # Sensordata: presence and per-step dim must agree. Adding sensordata
    # to a legacy (no-sd) dataset (or vice versa) would leave half the rows
    # without it, which downstream consumers can't slice cleanly — easier
    # to refuse and tell the user to re-collect with --force.
    ex_has_sd = "sensordata" in existing
    new_has_sd = new_nsensor > 0
    if ex_has_sd != new_has_sd:
        mismatches.append(
            f"sensordata presence: cached={ex_has_sd} vs current={new_has_sd}"
        )
    elif ex_has_sd:
        ex_nsensor = int(existing["sensordata"].shape[-1])
        if ex_nsensor != new_nsensor:
            mismatches.append(
                f"nsensor: cached={ex_nsensor} vs current={new_nsensor}"
            )

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

    # Load MPPI cfg first when using Warp — the GPU env needs `nworld=cfg.K`
    # (the rollout-batch width) baked in at construction time and `nworld`
    # is fixed for the env's lifetime. Pre-CPU path is unchanged.
    mppi_cfg = MPPIConfig.load(args.env)
    if args.warp:
        env = make_env(args.env, use_warp=True, nworld=mppi_cfg.K)
        print(f"[warp] using GPU rollout for K={mppi_cfg.K} sample trajectories")
    else:
        env = make_env(args.env)

    # Probe sensordata presence once on the freshly-reset env. If non-empty,
    # we'll allocate a per-step buffer and capture each step's sensordata
    # alongside obs/action/cost. Envs without it (gym_wrapper, anything
    # without a MuJoCo model) just skip the buffer entirely.
    env.reset()
    sample_sd = _try_capture_sensordata(env)
    nsensor = sample_sd.size if sample_sd is not None else 0

    # Validate append compatibility now that env + mppi_cfg + sensordata
    # presence are all known.
    existing_M = 0
    if existing is not None:
        _validate_append_compat(existing, args, env, mppi_cfg, nsensor)
        existing_M = int(existing["attrs"].get("M", existing["states"].shape[0]))

    controller = MPPI(env, cfg=mppi_cfg)

    obs_dim = env.obs_dim
    act_dim = env.action_dim
    M, T = args.M, args.T

    # Per-rollout buffers (one trajectory at a time — incremental writes
    # mean we no longer hold all M in RAM). Reused each loop iteration.
    states_row = np.zeros((T, obs_dim), dtype=np.float32)
    actions_row = np.zeros((T, act_dim), dtype=np.float32)
    costs_row = np.zeros((T,), dtype=np.float32)
    sensordata_row = (
        np.zeros((T, nsensor), dtype=np.float32) if nsensor > 0 else None
    )

    sd_msg = f", nsensor={nsensor}" if sensordata_row is not None else ""
    if existing is not None:
        print(f"appending {M} rollouts to existing {existing_M} on {args.env}  "
              f"(obs_dim={obs_dim}, act_dim={act_dim}, "
              f"auto_reset={args.auto_reset}{sd_msg})")
    else:
        print(f"collecting {M} rollouts x {T} steps on {args.env}  "
              f"(obs_dim={obs_dim}, act_dim={act_dim}, "
              f"auto_reset={args.auto_reset}{sd_msg})")
    print(f"→ {out_path}  (incremental: each rollout flushed on completion)")

    # ---- Initialise the on-disk file with resizable datasets ----
    # Fresh: empty resizable datasets at size 0, will grow per rollout.
    # Append: rewrite the existing rows into resizable datasets first
    #         (a one-time migration cost; legacy files weren't created
    #         with `maxshape=None` so we can't extend them in place),
    #         then append new rollouts. After this prologue the file
    #         is "incrementally extensible" forever.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_dataset(f, name, initial_data, full_shape, dtype=np.float32):
        """Helper: create a resizable dataset. `initial_data` may be None
        (size-0 placeholder) or a numpy array (seed with existing rows).
        ``full_shape`` is (None, *trailing) — trailing dims are fixed."""
        if initial_data is None:
            f.create_dataset(
                name,
                shape=(0,) + tuple(full_shape[1:]),
                maxshape=full_shape,
                chunks=True,
                dtype=dtype,
            )
        else:
            f.create_dataset(
                name,
                data=initial_data.astype(dtype),
                maxshape=full_shape,
                chunks=True,
            )

    with h5py.File(out_path, "w") as f:
        _create_dataset(
            f, "states",
            existing["states"] if existing is not None else None,
            (None, T, obs_dim),
        )
        _create_dataset(
            f, "actions",
            existing["actions"] if existing is not None else None,
            (None, T, act_dim),
        )
        _create_dataset(
            f, "costs",
            existing["costs"] if existing is not None else None,
            (None, T),
        )
        if sensordata_row is not None:
            _create_dataset(
                f, "sensordata",
                existing["sensordata"] if (existing is not None and "sensordata" in existing) else None,
                (None, T, nsensor),
            )
        # Static attrs — set once, frozen for the lifetime of the file.
        # `M` is updated incrementally inside the rollout loop below.
        f.attrs["env"] = args.env
        f.attrs["M"] = existing_M
        f.attrs["T"] = T
        f.attrs["obs_dim"] = obs_dim
        f.attrs["act_dim"] = act_dim
        f.attrs["seed"] = args.seed
        f.attrs["auto_reset"] = args.auto_reset
        f.attrs["mppi_cfg"] = json.dumps(asdict(mppi_cfg))

    # ---- Rollout loop with incremental writes ----
    # Open in r+ for the duration: we resize + write a new row + flush
    # after each completed rollout, so a SIGKILL / OOM / wallclock cap at
    # any point salvages all rollouts up to the last flush.
    outer = tqdm(range(M), desc="rollouts", unit="traj")
    new_costs_so_far: list[float] = []
    with h5py.File(out_path, "r+") as f:
        for i in outer:
            # Deterministic per-trajectory init so reruns are reproducible.
            # Offset by existing_M when appending so new rollouts don't duplicate
            # the seeds that produced the cached trajectories.
            np.random.seed(args.seed + existing_M + i)
            env.reset()
            controller.reset()

            # Reset row buffers to zero so prior trailing data from a shorter
            # auto-reset segment doesn't leak across rollouts.
            states_row.fill(0.0)
            actions_row.fill(0.0)
            costs_row.fill(0.0)
            if sensordata_row is not None:
                sensordata_row.fill(0.0)

            for t in range(T):
                obs = env._get_obs()
                state = env.get_state()
                # Capture sensordata BEFORE plan_step / step so it pairs with
                # the (obs, state) at time t (i.e. pre-action). plan_step does
                # batched rollouts that mutate scratch buffers but not the
                # live env.data; step() advances to t+1, which is the wrong
                # alignment for the saved row.
                if sensordata_row is not None:
                    sensordata_row[t] = env.data.sensordata
                action, info = controller.plan_step(state)
                _, cost, done, _ = env.step(action)

                states_row[t] = obs
                actions_row[t] = action
                costs_row[t] = cost

                if done and t < T - 1:
                    if args.auto_reset:
                        env.reset()
                        controller.reset()
                        continue
                    # no auto-reset: fill the remainder with the last (obs, 0, 0).
                    # Downstream BC drops trailing zeros via cost/action sanity, but
                    # for the common non-terminating envs we just never hit this.
                    break

            # ---- Append this rollout to the file + flush ----
            slot = existing_M + i
            f["states"].resize(slot + 1, axis=0)
            f["states"][slot] = states_row
            f["actions"].resize(slot + 1, axis=0)
            f["actions"][slot] = actions_row
            f["costs"].resize(slot + 1, axis=0)
            f["costs"][slot] = costs_row
            if sensordata_row is not None:
                f["sensordata"].resize(slot + 1, axis=0)
                f["sensordata"][slot] = sensordata_row
            # Update the rollout count attr each iter so a partial file
            # is self-describing — readers can use attrs["M"] without
            # also checking dataset.shape[0].
            f.attrs["M"] = slot + 1
            # flush() forces buffered writes to OS; fsync would be needed
            # for hard crash safety against OS caches, but flush+normal
            # exit covers the common kill / Ctrl-C / OOM paths cheaply.
            f.flush()

            total_cost = float(costs_row.sum())
            new_costs_so_far.append(total_cost)
            outer.set_postfix(total_cost=f"{total_cost:.1f}",
                              last_cost_min=f"{info['cost_min']:.2f}")

    total_M = existing_M + M
    if existing is not None:
        new_cost_mean = float(np.mean(new_costs_so_far)) if new_costs_so_far else float("nan")
        # Read the full costs back to get the all-time mean (cheap; T*total_M floats).
        with h5py.File(out_path, "r") as f:
            all_cost_mean = float(f["costs"][:].sum(axis=1).mean())
        print(f"\nappended {M} trajectories (new mean cost: {new_cost_mean:.2f}); "
              f"dataset now holds {total_M} trajectories of length {T} at {out_path}")
        print(f"mean total cost across ALL trajectories: {all_cost_mean:.2f}")
    else:
        cost_mean = float(np.mean(new_costs_so_far)) if new_costs_so_far else float("nan")
        print(f"\nsaved {M} trajectories of length {T} to {out_path}")
        print(f"mean total cost across trajectories: {cost_mean:.2f}")
    env.close()


if __name__ == "__main__":
    main()
