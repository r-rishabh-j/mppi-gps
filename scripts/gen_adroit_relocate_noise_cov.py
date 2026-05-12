"""Generate a hand-synergy correlated noise covariance for adroit_relocate.

Off-diagonals encode two priors:
  - hand synergy: positive correlation between every finger joint pair;
  - lift coupling: positive correlation between arm-z and each finger.
Per-dim marginal std stays ``noise_sigma · env.noise_scale``, so setting
both correlations to 0 recovers the diagonal default exactly.

Default output: ``configs/adroit_relocate_correlated.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.envs import make_env


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "configs" / "adroit_relocate_correlated.json"

# Adroit actuator layout: 0-2 arm slides, 3-5 arm rot, 6-7 wrist,
# 8-11 ff, 12-15 mf, 16-19 rf, 20-24 lf, 25-29 th.
ARM_Z_IDX = 2
FINGER_IDX = list(range(8, 30))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(_DEFAULT_OUT),
                   help="output config path.")
    p.add_argument("--noise-sigma", type=float, default=0.25,
                   help="Scalar multiplier on env.noise_scale per dim.")
    p.add_argument("--hand-synergy", type=float, default=0.40,
                   help="Pairwise rho between finger joints (8-29).")
    p.add_argument("--lift-coupling", type=float, default=0.25,
                   help="Rho between arm-z (idx 2) and each finger joint.")
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--lam", type=float, default=0.08)
    p.add_argument("--adaptive-lam", action="store_true", default=False)
    p.add_argument("--n-eff-threshold", type=float, default=256)
    p.add_argument("--open-loop-steps", type=int, default=1)
    return p.parse_args()


def build_correlation(
    nu: int,
    hand_synergy: float,
    lift_coupling: float,
) -> np.ndarray:
    """Correlation matrix; combine with per-dim std for covariance."""
    C = np.eye(nu)
    for i in FINGER_IDX:
        for j in FINGER_IDX:
            if i != j:
                C[i, j] = hand_synergy
    for f in FINGER_IDX:
        C[ARM_Z_IDX, f] = lift_coupling
        C[f, ARM_Z_IDX] = lift_coupling
    return C


def main() -> None:
    args = parse_args()

    env = make_env("adroit_relocate")
    nu = env.action_dim

    sigma_per_dim = args.noise_sigma * np.asarray(env.noise_scale, dtype=np.float64)
    assert sigma_per_dim.shape == (nu,)

    C = build_correlation(nu, args.hand_synergy, args.lift_coupling)
    D = np.diag(sigma_per_dim)
    cov = D @ C @ D                          # (nu, nu)

    # Validate PD before writing — `noise_cov` consumers will reject a
    # non-PD matrix at MPPI construction, but failing here gives a
    # clearer message tied to the parameter choice.
    eigs = np.linalg.eigvalsh(cov)
    if eigs.min() <= 0:
        raise SystemExit(
            f"resulting cov is not positive-definite: min eig = {eigs.min():.6e}\n"
            f"  reduce --hand-synergy ({args.hand_synergy}) or "
            f"--lift-coupling ({args.lift_coupling}) and try again."
        )

    # Sanity report
    print(f"adroit_relocate noise covariance:")
    print(f"  noise_sigma     = {args.noise_sigma}")
    print(f"  hand_synergy    = {args.hand_synergy}  "
          f"(off-diagonal correlation among finger joints, idx 8-29)")
    print(f"  lift_coupling   = {args.lift_coupling}  "
          f"(correlation between arm-z idx 2 and each finger joint)")
    print(f"  cov shape       = {cov.shape}")
    print(f"  min eigenvalue  = {eigs.min():.6e}  (PD check: OK)")
    print(f"  max abs corr    = {float(np.abs(cov / np.outer(sigma_per_dim, sigma_per_dim) - np.eye(nu)).max()):.3f}")
    print(f"  per-dim sigma   = noise_sigma * env.noise_scale "
          f"(byte-identical marginals to diagonal config)")

    payload = {
        "K": args.K,
        "H": args.H,
        "lam": args.lam,
        # Kept for documentation; ignored by MPPI when noise_cov is set.
        "noise_sigma": args.noise_sigma,
        "adaptive_lam": args.adaptive_lam,
        "n_eff_threshold": args.n_eff_threshold,
        "open_loop_steps": args.open_loop_steps,
        # Generator metadata so future readers can regenerate / re-tune.
        "_generated_by": "scripts/gen_adroit_relocate_noise_cov.py",
        "_hand_synergy": args.hand_synergy,
        "_lift_coupling": args.lift_coupling,
        "noise_cov": cov.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
