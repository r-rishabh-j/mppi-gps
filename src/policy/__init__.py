"""Policy package — shared toggles + helpers used by both policy classes."""

# ---------------------------------------------------------------------------
# Action-space output normalization (toggle)
# ---------------------------------------------------------------------------
#
# When True, the policy network's action head outputs in *normalized* action
# space (per-dim ∈ ~[-1, 1]) and we apply a per-dim affine `a = scale·n + bias`
# at the policy boundary (`act_np`, `action()`) to recover the physical action.
# Inversely, training labels (which are physical actions from MPPI) are
# normalized before the loss is computed against the network output. This
# equalizes per-dim gradient contribution across heterogeneous action ranges
# — the motivating case is Adroit Relocate, where arm slides span ~0.2 m and
# fingers span ~2.0 rad (10× ratio). Without this normalization, an unweighted
# MSE / mean-distance loss is dominated by finger residuals and arm dims
# receive proportionally tiny gradients, producing a policy whose arm
# commands stay close to current joint positions → tiny actuator forces →
# the visible "arm too weak to carry" failure mode.
#
# When False, behavior is **byte-identical** to pre-normalization
# checkpoints: no `_act_scale` / `_act_bias` buffers are registered, the
# network output flows straight through `act_np`'s clip, and old `.pt`
# files load with `load_state_dict(strict=True)` exactly as before. Flip
# off when you need to load a checkpoint trained before this change.
#
# IMPORTANT: the toggle must match the value used at training time when
# loading a checkpoint. A model trained with USE_ACT_NORM=True has its
# network learning a normalized output distribution; loading those weights
# under =False would skip the affine and produce wrong-magnitude actions.
# A model trained with =False has no `_act_scale` / `_act_bias` buffers in
# its state_dict; loading under =True would fail strict load with missing
# keys. Either case is loud rather than silently wrong, which is the point.
USE_ACT_NORM = True


# ---------------------------------------------------------------------------
# Hand-crafted input featurization (port of upstream `featurize_obs`)
# ---------------------------------------------------------------------------
#
# Enabled via `PolicyConfig.featurize = "hand_crafted"`. The transform is
# env-shape-aware:
#
#   * 4-D raw obs (acrobot ``[θ1, θ2, ω1, ω2]``):
#       → ``[sin θ1, cos θ1, sin θ2, cos θ2, ω1/2.5, ω2/5.0]`` (6-D)
#     Encodes angle as a 2π-periodic feature so the network doesn't have
#     to learn the wrap-around, plus per-dim velocity scaling to ≈[-1, 1].
#
#   * 6-D raw obs (point_mass ``[pos (2), vel (2), goal (2)]``):
#       → ``[(pos - goal)/0.29, vel/0.5, goal/0.29]`` (6-D)
#     Explicit ``(pos - goal)`` channel — the network gets the
#     goal-reaching delta as input rather than having to learn it from
#     ``pos`` and ``goal`` separately. All channels normalized to ≈[-1, 1].
#
#   * Anything else → identity (no-op). The training loop's existing
#     `RunningNormalizer` is bypassed in this branch — the featurizer
#     already produces unit-scaled input.
#
# Constants 0.29 / 0.5 / 2.5 / 5.0 are taken verbatim from upstream
# (`jaiselsingh1/mppi-gps`) and reflect the typical magnitudes for those
# tasks. They are NOT learned — if you change the env (e.g. retune
# point_mass with a larger arena), edit these or fall back to
# `featurize="running_norm"`.

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
    # Unknown obs_dim — pass through. RunningNormalizer (when enabled)
    # downstream will still see this and apply per-dim affine.
    return obs


def featurized_dim(raw_obs_dim: int) -> int:
    """Output dim of `featurize_obs` for a given raw input dim. Used to size
    the first MLP layer at construction time when `featurize="hand_crafted"`.
    """
    if raw_obs_dim == 4:
        return 6
    return raw_obs_dim   # 6-D point_mass passes through at same dim; others identity
