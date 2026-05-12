"""Policy package: shared toggles + featurization helpers."""

# Action-space output normalization. True → network head outputs in
# normalized space (~[-1, 1]); per-dim affine maps to physical at the
# policy boundary. Critical for envs with heterogeneous action ranges.
# Must match between training and eval (loud failure either way).
USE_ACT_NORM = True


# Hand-crafted input featurization (selected by
# ``PolicyConfig.featurize="hand_crafted"``):
#
# * 4-D raw obs (acrobot [θ1,θ2,ω1,ω2]) → 6-D
#   [sin θ1, cos θ1, sin θ2, cos θ2, ω1/2.5, ω2/5.0]
# * 6-D raw obs (point_mass [pos,vel,goal]) → 6-D
#   [(pos-goal)/0.29, vel/0.5, goal/0.29]
# * Anything else → identity. RunningNormalizer is bypassed.

import torch
from jaxtyping import Float
from torch import Tensor


def featurize_obs(obs: Float[Tensor, "*batch obs_dim"]) -> Float[Tensor, "*batch out_dim"]:
    """Hand-crafted feature transform; see module docstring."""
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
