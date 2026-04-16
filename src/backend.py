"""Backend abstraction for CPU/GPU execution.

Provides a single config knob (`backend: "cpu"` or `"gpu"`) that controls
whether MPPI rollouts and cost computation run on CPU (numpy + MuJoCo C API)
or GPU (JAX + MJX).  JAX handles Metal vs CUDA automatically — no
platform-specific code is needed.
"""

import enum
from typing import Any


class Backend(enum.Enum):
    CPU = "cpu"
    GPU = "gpu"


def get_backend(name: str = "cpu") -> Backend:
    """Parse a backend string into the enum."""
    return Backend(name.lower())


def get_device() -> Any:
    """Return the best available JAX device (GPU/Metal/CUDA, or CPU fallback).

    Returns a ``jax.Device`` object.  On Apple Silicon with jax-metal
    installed this will be a Metal device; on NVIDIA with jax[cuda] it
    will be a CUDA device; otherwise CPU.
    """
    import jax

    for d in jax.devices():
        if d.platform != "cpu":
            return d
    return jax.devices("cpu")[0]


def ensure_gpu_available() -> Any:
    """Assert that a non-CPU JAX device exists.  Returns the device.

    Raises ``RuntimeError`` with install hints when no GPU is found.
    """
    import jax

    gpu_devices = [d for d in jax.devices() if d.platform != "cpu"]
    if not gpu_devices:
        raise RuntimeError(
            f"No GPU device found. Available JAX devices: {jax.devices()}. "
            "Install jax-metal (Apple Silicon) or jax[cuda12] (NVIDIA)."
        )
    return gpu_devices[0]
