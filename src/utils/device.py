"""Device selection helper for torch training.

Resolves "auto" to cuda → mps → cpu, or honours an explicit override.
"""
import torch


def pick_device(preferred: str | None = "auto") -> torch.device:
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    p = preferred.lower()
    if p == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        return torch.device("cuda")
    if p == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("mps requested but not available")
        return torch.device("mps")
    if p == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device '{preferred}'")
