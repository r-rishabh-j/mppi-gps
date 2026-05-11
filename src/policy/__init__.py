"""Policy package — shared toggles + helpers used by both policy classes."""

# Action-space output normalization. When True, the network's action head
# outputs in normalized action space (~[-1, 1] per dim) and a per-dim
# affine `a = scale·n + bias` maps to physical at the policy boundary
# (`act_np`, `action`). Equalizes per-dim gradient contribution across
# heterogeneous action ranges — critical for Adroit Relocate where arm
# slides (~0.2 m) and finger joints (~2.0 rad) differ 10×.
#
# Must match between training and eval: a model trained under True has
# `_act_scale`/`_act_bias` buffers; loading under False would fail strict
# load (missing keys). A model trained under False would silently produce
# wrong-magnitude actions under True. Either case is loud rather than
# silently wrong.
USE_ACT_NORM = True


# Hand-crafted input featurization (port of upstream `featurize_obs`).
# Enabled via `PolicyConfig.featurize = "hand_crafted"`. Env-shape-aware:
#
#   * 4-D raw obs (acrobot `[θ1, θ2, ω1, ω2]`):
#       → `[sin θ1, cos θ1, sin θ2, cos θ2, ω1/2.5, ω2/5.0]` (6-D)
#     Wraps the angle into a 2π-periodic feature; per-dim vel scaled ≈[-1, 1].
#
#   * 6-D raw obs (point_mass `[pos (2), vel (2), goal (2)]`):
#       → `[(pos - goal)/0.29, vel/0.5, goal/0.29]` (6-D)
#     Explicit delta-to-goal channel; all normalized to ≈[-1, 1].
#
#   * Anything else → identity. RunningNormalizer is bypassed in this mode.
#
# Constants 0.29 / 0.5 / 2.5 / 5.0 are upstream's, reflecting typical
# magnitudes. Edit or fall back to "running_norm" if you change the env.

import torch
from jaxtyping import Float
from torch import Tensor


def featurize_obs(obs: Float[Tensor, "*batch obs_dim"]) -> Float[Tensor, "*batch out_dim"]:
    """Hand-crafted feature transform. See module docstring for the schema."""
    raw_dim = obs.shape[-1]
    if raw_dim == 6:
        new_obs = torch.empty(*obs.shape[:-1], 6, device=obs.device, dtype=obs.dtype)
        pos = obs[..., 0:2]
        vel = obs[..., 2:4]
        goal = obs[..., 4:6]
        new_obs[..., 0:2] = (pos - goal) / 0.29
        new_obs[..., 2:4] = vel / 0.5
        new_obs[..., 4:6] = goal / 0.29
        return new_obs
    if raw_dim == 4:
        new_obs = torch.empty(*obs.shape[:-1], 6, device=obs.device, dtype=obs.dtype)
        new_obs[..., 0] = torch.sin(obs[..., 0])
        new_obs[..., 1] = torch.cos(obs[..., 0])
        new_obs[..., 2] = torch.sin(obs[..., 1])
        new_obs[..., 3] = torch.cos(obs[..., 1])
        new_obs[..., 4] = obs[..., 2] / 2.5
        new_obs[..., 5] = obs[..., 3] / 5.0
        return new_obs
    return obs


def featurized_dim(raw_obs_dim: int) -> int:
    """Output dim of `featurize_obs` for a given raw input dim."""
    if raw_obs_dim == 4:
        return 6
    return raw_obs_dim
