"""Generate a hand-synergy correlated noise covariance for adroit_relocate.

Encodes three priors that pure per-dim independent noise rarely samples:

1. **Hand synergy** — the 22 finger joints (idx 8-29) are positively
   correlated (default rho = 0.40 between any pair). A single MPPI noise
   sample tends to open or close all fingers together — power-grasp
   inductive bias. Without this, MPPI would need O(2^22) samples to
   stumble into "all fingers closed at the same moment".

2. **Lift coupling** — arm-z (idx 2, A_ARTz, the slide that drives the
   hand up/down) is positively correlated (default rho = 0.25) with
   every finger joint. Single sample tends to explore "lift hand + grip
   tighter" or "lower hand + open grip" together — the two coordination
   modes critical to the lift transition. Adroit's "stops moving after
   grasp" failure mode partly comes from arm-z and fingers being
   sampled independently, so the joint "lift now while keeping grip"
   trajectory is rarely synthesised.

3. **Everything else independent** — arm rotation (3-5), wrist (6-7),
   arm xy (0-1) stay diagonal. These dims genuinely need independent
   exploration; baking in correlation here would constrain MPPI for no
   good reason.

Per-dim marginal std stays identical to the diagonal baseline:
``sigma_per_dim = noise_sigma * env.noise_scale``. Only the joint
(off-diagonal) distribution changes, so the cov-matrix path is a
strict superset of the diagonal default — turning correlations to 0
recovers the legacy behaviour exactly.

Output: ``configs/adroit_relocate_correlated.json`` by default. Pass
``--out`` to overwrite ``adroit_relocate_best.json`` instead.

Examples:
    python -m scripts.gen_adroit_relocate_noise_cov            # default rhos
    python -m scripts.gen_adroit_relocate_noise_cov --hand-synergy 0.5 \\
        --lift-coupling 0.3 --noise-sigma 0.2
    python -m scripts.gen_adroit_relocate_noise_cov --out configs/adroit_relocate_best.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.envs import make_env


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "configs" / "adroit_relocate_correlated.json"

# Adroit relocate actuator layout (from env.model.actuator_name):
#   0-2  : A_ARTx, A_ARTy, A_ARTz   — arm slides
#   3-5  : A_ARRx, A_ARRy, A_ARRz   — arm rotation
#   6-7  : A_WRJ1, A_WRJ0           — wrist
#   8-11 : A_FFJ3..FFJ0             — forefinger (knuckle, prox, mid, dist)
#   12-15: A_MFJ3..MFJ0             — middle finger
#   16-19: A_RFJ3..RFJ0             — ring finger
#   20-24: A_LFJ4..LFJ0             — little finger (5 joints incl metacarpal)
#   25-29: A_THJ4..THJ0             — thumb
ARM_Z_IDX = 2
FINGER_IDX = list(range(8, 30))   # all 22 finger joints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(_DEFAULT_OUT),
                   help="output config path (default: "
                        "configs/adroit_relocate_correlated.json). "
                        "Pass configs/adroit_relocate_best.json to overwrite "
                        "the env's default config.")
    p.add_argument("--noise-sigma", type=float, default=0.25,
                   help="scalar multiplier on env.noise_scale for the per-dim "
                        "marginal std (matches the diagonal-config field of the "
                        "same name). Default 0.25 — same as adroit_relocate_best.json.")
    p.add_argument("--hand-synergy", type=float, default=0.40,
                   help="Pairwise correlation between any two finger joints "
                        "(idx 8-29). 0.0 = independent (diagonal). 0.7 = "
                        "strong synergy (loses ~half of independent exploration). "
                        "Default 0.40.")
    p.add_argument("--lift-coupling", type=float, default=0.25,
                   help="Pairwise correlation between arm-z (idx 2) and each "
                        "finger joint. Encodes the 'lift+grip together' bias. "
                        "0.0 = independent. Default 0.25.")
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
    """Build the (nu, nu) correlation matrix C with C[i, i] = 1, off-diagonal
    entries set per the design above. Caller is responsible for combining
    with per-dim std to get covariance.
    """
    C = np.eye(nu)
    # Hand synergy: every pair of finger joints
    for i in FINGER_IDX:
        for j in FINGER_IDX:
            if i != j:
                C[i, j] = hand_synergy
    # Lift coupling: arm-z paired with each finger joint
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
