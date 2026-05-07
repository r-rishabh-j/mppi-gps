"""Load a saved policy checkpoint as an MPPI prior for the standalone
``scripts/run_*.py`` viewers.

Two-function surface — drop-in for any MPPI run script:

    from src.utils.policy_prior_loader import (
        add_policy_prior_args, resolve_policy_prior,
    )

    parser = argparse.ArgumentParser()
    add_policy_prior_args(parser)          # adds --policy-ckpt / --alpha / --prior-type
    args = parser.parse_args()
    ...
    env = make_env(...)
    controller = MPPI(env, cfg)
    prior_fn = resolve_policy_prior(args, env)   # None when --policy-ckpt is omitted
    ...
    action, info = controller.plan_step(state, prior=prior_fn)

When ``--policy-ckpt`` is omitted (default), MPPI runs vanilla — bit-for-bit
identical to the existing scripts. When set, the policy is loaded
auto-detecting class (Gaussian vs Deterministic) from the checkpoint /
sibling ``config.json``, and ``make_policy_prior`` from the GPS trainer is
used to build the same prior shape (``nll`` for Gaussian, ``mean_distance``
for Deterministic) the C-step would have produced. Useful for visualising
how an α > 0 rollout looks at viewer-time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

from src.envs.base import BaseEnv
from src.gps.mppi_gps_unified import make_policy_prior
from src.policy.deterministic_policy import DeterministicPolicy
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import PolicyConfig
from src.utils.experiment import load_checkpoint


def add_policy_prior_args(p: argparse.ArgumentParser) -> None:
    """Add ``--policy-ckpt`` / ``--alpha`` / ``--prior-type`` to ``p``.

    All three are optional; behaviour reverts to vanilla MPPI when
    ``--policy-ckpt`` is unset (or ``--alpha`` is 0). Safe to call on
    parsers that already define ``--seed`` etc. — these three names
    don't collide with anything in the existing ``run_*.py`` scripts.
    """
    p.add_argument(
        "--policy-ckpt", default=None,
        help="Path to a wrapped .pt checkpoint OR a run dir (uses best.pt). "
             "When set, MPPI uses this policy as a soft prior with weight "
             "--alpha. Auto-detects policy class from the checkpoint / "
             "sibling config.json.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.0,
        help="Policy-prior weight (default 0 = vanilla MPPI). Same scale "
             "as GPSConfig.policy_augmented_alpha; typical 0.01–0.1.",
    )
    p.add_argument(
        "--prior-type", default="auto",
        choices=["auto", "nll", "mean_distance"],
        help="Prior shape. 'auto' (default) picks 'nll' for Gaussian, "
             "'mean_distance' for Deterministic. Override only if you "
             "explicitly want mean_distance on a Gaussian checkpoint "
             "(e.g. to match a GPS run that used --policy-prior=mean_distance).",
    )


def _resolve_ckpt_and_config(ckpt_arg: str) -> tuple[Path, dict | None]:
    """Mirrors ``scripts/eval_checkpoint._resolve_ckpt_and_config``: accept
    either a single .pt file or a run dir (uses ``best.pt``). Returns the
    resolved path and the parsed sibling ``config.json`` if present."""
    p = Path(ckpt_arg)
    if p.is_dir():
        best = p / "best.pt"
        if not best.exists():
            raise FileNotFoundError(
                f"{best} not found; pass a specific .pt file instead."
            )
        cfg_path = p / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
        return best, cfg
    cfg_path = p.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    return p, cfg


def resolve_policy_prior(
    args: argparse.Namespace,
    env: BaseEnv,
    device: str | torch.device = "cpu",
) -> Callable | None:
    """Build the MPPI ``prior_fn`` from CLI flags. Returns ``None`` (→
    vanilla MPPI) when ``--policy-ckpt`` is unset or ``--alpha <= 0``."""
    ckpt_arg = getattr(args, "policy_ckpt", None)
    alpha = float(getattr(args, "alpha", 0.0))
    if not ckpt_arg or alpha <= 0.0:
        return None

    ckpt_path, run_cfg = _resolve_ckpt_and_config(ckpt_arg)
    blob = load_checkpoint(ckpt_path, map_location=device)
    policy_class = (
        blob.get("policy_class") or (run_cfg or {}).get("policy_class")
    )
    use_det = (policy_class == "DeterministicPolicy")

    # Use the same per-env PolicyConfig the trainers use, so a checkpoint
    # trained with bumped hidden_dims (e.g. adroit's 512x3) loads cleanly.
    env_name = (run_cfg or {}).get("env") or type(env).__name__.lower()
    policy_cfg = PolicyConfig.for_env(env_name)

    PolicyCls = DeterministicPolicy if use_det else GaussianPolicy
    policy = PolicyCls(
        env.obs_dim, env.action_dim, policy_cfg,
        device=device, action_bounds=env.action_bounds,
    )
    policy.load_state_dict(blob["state_dict"])
    # Same NaN-weights guard as eval_checkpoint: a poisoned checkpoint
    # silently emits NaN actions and freezes the env at reset; raise loudly.
    bad = sorted(
        k for k, v in policy.state_dict().items()
        if not torch.isfinite(v).all()
    )
    if bad:
        raise ValueError(
            f"checkpoint {ckpt_path} has non-finite weights in {len(bad)} "
            f"tensors (first 5: {bad[:5]}). Retrain — checkpoint is "
            f"unrecoverable."
        )
    policy.eval()

    prior_type = args.prior_type
    if prior_type == "auto":
        prior_type = "mean_distance" if use_det else "nll"
    if prior_type == "nll" and use_det:
        raise ValueError(
            "--prior-type=nll requires a Gaussian checkpoint "
            "(DeterministicPolicy has no log_prob)."
        )

    print(
        f"[policy prior] loaded {ckpt_path}  "
        f"class={'Det' if use_det else 'Gauss'}  "
        f"prior={prior_type}  alpha={alpha}"
    )
    return make_policy_prior(policy, env, alpha, prior_type)
