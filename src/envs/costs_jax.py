"""Pure JAX cost functions for all environments.

Each function is stateless (no ``self``) so it can be JIT-compiled and
vmap-ed inside ``MJXEnv.batch_rollout``.  Environment-specific constants
(goal position, control weight, etc.) are passed as arguments.

The numpy originals in each environment subclass remain untouched — these
JAX versions are only used by the GPU (MJX) path.
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared helpers — mirror MuJoCoEnv.state_qpos / state_qvel
# ---------------------------------------------------------------------------

def state_qpos(states, nq: int):
    """Extract qpos from full physics state.  states[..., 1:1+nq]."""
    return states[..., 1:1 + nq]


def state_qvel(states, nq: int, nv: int):
    """Extract qvel from full physics state.  states[..., 1+nq:1+nq+nv]."""
    start = 1 + nq
    return states[..., start:start + nv]


# ---------------------------------------------------------------------------
# PointMass
# ---------------------------------------------------------------------------

def point_mass_running_cost(states, actions, sensordata, *, goal, nq, nv):
    """(K,H,nstate), (K,H,nu), _ → (K,H)"""
    qpos = state_qpos(states, nq)
    qvel = state_qvel(states, nq, nv)
    pos_err = qpos - goal[None, None, :]
    pos_cost = jnp.sum(pos_err ** 2, axis=-1)
    vel_cost = 5.0 * jnp.sum(qvel * pos_err, axis=-1)
    return pos_cost + vel_cost


def point_mass_terminal_cost(states, sensordata, *, goal, nq, nv):
    """(K,nstate), _ → (K,)"""
    qpos = state_qpos(states, nq)
    qvel = state_qvel(states, nq, nv)
    pos_err = qpos - goal[None, :]
    return 0.0 * jnp.sum(pos_err ** 2, axis=-1) + 0.5 * jnp.sum(qvel ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Acrobot
#
# NOTE: The CPU version reads tip position from ``sensordata[:,:,:3]``
# (a ``framepos`` sensor).  In MJX we pass ``site_xpos`` instead, which
# is computed after each ``mjx.step`` + ``mjx.forward``.  The interface
# is the same: a (..., 3) array of the tip's world-frame position.
# ---------------------------------------------------------------------------

def acrobot_running_cost(states, actions, site_xpos, *, target, margin, nq, nv):
    """(K,H,nstate), (K,H,nu), (K,H,3) → (K,H)

    ``site_xpos`` is the 3-D position of the acrobot tip site.
    """
    dist = jnp.linalg.norm(site_xpos - target[None, None, :], axis=-1)
    return (dist / margin) + 2.0 * (4.0 - jnp.linalg.norm(site_xpos, axis=-1))


def acrobot_terminal_cost(states, site_xpos, *, target, w_terminal, nq, nv):
    """(K,nstate), (K,3) → (K,)"""
    dist = jnp.linalg.norm(site_xpos - target[None, :], axis=-1)
    qvel = state_qvel(states, nq, nv)
    vel_cost = jnp.sum(qvel ** 2, axis=-1)
    return w_terminal * (dist + 5.0 * vel_cost)


# ---------------------------------------------------------------------------
# HalfCheetah
# ---------------------------------------------------------------------------

def half_cheetah_running_cost(states, actions, sensordata, *, nq, nv,
                              w_vel=1.0, w_pitch=0.5, w_controls=0.001):
    """(K,H,nstate), (K,H,nu), _ → (K,H)"""
    qpos = state_qpos(states, nq)
    qvel = state_qvel(states, nq, nv)
    vx = qvel[:, :, 0]
    torso_pitch = qpos[:, :, 2]
    ctrl_cost = jnp.sum(jnp.square(actions), axis=-1)
    return -w_vel * vx + w_pitch * (torso_pitch ** 2) + w_controls * ctrl_cost


def half_cheetah_terminal_cost(states, sensordata):
    """(K,nstate), _ → (K,)"""
    return jnp.zeros(states.shape[0])


# ---------------------------------------------------------------------------
# Hopper
# ---------------------------------------------------------------------------

def hopper_running_cost(states, actions, sensordata, *, nq, nv,
                        fwd_w=1.0, healthy_reward=1.0, ctrl_w=0.001,
                        z_min=0.7, angle_max=0.2):
    """(K,H,nstate), (K,H,nu), _ → (K,H)"""
    qpos = state_qpos(states, nq)
    qvel = state_qvel(states, nq, nv)
    vx = qvel[:, :, 0]
    ctrl_cost = jnp.sum(jnp.square(actions), axis=-1)
    z = qpos[:, :, 1]       # rootz
    angle = qpos[:, :, 2]   # rooty
    healthy = (z > z_min) & (jnp.abs(angle) < angle_max)
    healthy = healthy.astype(jnp.float32)
    return -fwd_w * vx - healthy_reward * healthy + ctrl_w * ctrl_cost


def hopper_terminal_cost(states, sensordata):
    """(K,nstate), _ → (K,)"""
    return jnp.zeros(states.shape[0])
