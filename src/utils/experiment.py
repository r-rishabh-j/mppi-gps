"""Experiment bookkeeping: run dirs, config dumps, checkpoint wrapping.

Run dir layout:
    <base>/<YYYYmmdd-HHMMSS>_<env>_<name>/
        config.json   iter_<k>.pt   best.pt   final.pt   log.csv

Wrapped checkpoint: ``{state_dict, policy_class, round, ...}``.
``load_checkpoint`` also accepts legacy raw state_dicts.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.decode().strip()
    except Exception:
        return None


def make_run_dir(base: str | Path, env: str, name: str) -> Path:
    """Create experiments/<base>/<timestamp>_<env>_<name>/ and return the path."""
    safe_name = name.replace("/", "_").replace(" ", "_")
    run_dir = Path(base) / f"{_timestamp()}_{env}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _jsonable(obj: Any) -> Any:
    """Convert dataclasses / paths / tensors / torch.device to JSON-safe values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def write_config(run_dir: Path, payload: dict) -> Path:
    """Write config.json atomically (tmp + rename)."""
    path = run_dir / "config.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=False))
    tmp.replace(path)
    return path


def update_config(run_dir: Path, updates: dict) -> None:
    """Merge `updates` into config.json (top-level keys only)."""
    path = run_dir / "config.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(_jsonable(updates))
    path.write_text(json.dumps(data, indent=2, sort_keys=False))


def save_checkpoint(
    path: str | Path,
    policy: torch.nn.Module,
    **extras: Any,
) -> None:
    """Save a wrapped checkpoint: state_dict + policy class name + caller extras."""
    payload = {
        "state_dict": policy.state_dict(),
        "policy_class": type(policy).__name__,
        **extras,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: Any = None) -> dict:
    """Load a checkpoint. Returns {state_dict, policy_class?, ...}.

    Back-compat: if the file is a raw state_dict (legacy), wraps it as
    {"state_dict": blob} with no policy_class.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob
    return {"state_dict": blob}


def copy_as(src: Path, dst: Path) -> None:
    """Copy src → dst, replacing dst if present."""
    shutil.copyfile(src, dst)


def load_state_dict_into(
    target_policy: torch.nn.Module,
    blob: dict,
    *,
    strict: bool = True,
) -> dict:
    """Load ``blob['state_dict']`` into ``target_policy``.

    Auto-fix: Gaussian → Deterministic strips the log_sigma half of the
    final-layer head (rest is shape-identical). Deterministic → Gaussian
    refuses (no log_sigma weights to invent).

    Returns ``{"converted", "src_class", "msg"}``.
    """
    src_class = blob.get("policy_class") or "<unknown>"
    target_class = type(target_policy).__name__
    state = blob["state_dict"]
    target_state = target_policy.state_dict()

    if src_class in ("<unknown>", target_class):
        target_policy.load_state_dict(state, strict=strict)
        return {
            "converted": False,
            "src_class": src_class,
            "msg": f"loaded {src_class} → {target_class} weights directly",
        }

    if src_class == "GaussianPolicy" and target_class == "DeterministicPolicy":
        remapped: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for k, v in state.items():
            if k not in target_state:
                skipped.append(k)
                continue
            tshape = target_state[k].shape
            if v.shape == tshape:
                remapped[k] = v
            elif v.ndim >= 1 and v.shape[0] == 2 * tshape[0] and v.shape[1:] == tshape[1:]:
                # Head: keep the mu rows (first half).
                remapped[k] = v[: tshape[0]].clone()
            else:
                raise RuntimeError(
                    f"can't convert Gaussian → Deterministic for {k}: "
                    f"source shape {tuple(v.shape)} vs target {tuple(tshape)}"
                )
        target_policy.load_state_dict(remapped, strict=True)
        msg = "converted GaussianPolicy → DeterministicPolicy (dropped log_sigma)"
        if skipped:
            msg += f" (ignored {len(skipped)} extra keys: {skipped[:3]}...)"
        return {"converted": True, "src_class": src_class, "msg": msg}

    if src_class == "DeterministicPolicy" and target_class == "GaussianPolicy":
        raise RuntimeError(
            "cannot warm-start GaussianPolicy from DeterministicPolicy: "
            "no log_sigma weights in the source."
        )

    raise RuntimeError(
        f"unsupported policy class transition: {src_class} → {target_class}"
    )
