import mujoco 
import numpy as np 
import matplotlib.pyplot as plt 

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI 
from src.utils.config import MPPIConfig

def build_cost_map(env, res = 200):
    theta = np.linspace(-np.pi, np.pi, res)
    T1, T2 = np.meshgrid(theta, theta)
    C = np.zeros_like(T1) 

    target = np.array([0.0, 0.0, 4.0])
    target_radius = 0.20
    margin = 4.0
    scale = np.sqrt(-2.0 * np.log(0.1)) 

    for i in range(res):
        for j in range(res):
            env.data.qpos[:] = [T1[i, j], T2[i, j]]
            env.data.qvel[:] = 0.0 
            mujoco.mj_forward(env.model, env.data)
            tip = env.data.site("tip").xpos.copy()

            dist = np.linalg.norm(tip - target)
            d_beyond = max(dist - target_radius, 0.0)
            reward = 1.0 if dist <= target_radius else np.exp(-0.5 * (d_beyond * scale / margin) ** 2)
            C[i, j] = 1.0 - reward
    return T1, T2, C 

env = Acrobot()
cfg = MPPIConfig(
        K=1000,
        H=174,
        lam=0.00010000000,
        noise_sigma=0.11239984567550243,
        adaptive_lam=False,
    )
controller = MPPI(env, cfg)

# precompute cost grid (using env's model/data, then reset after)
T1, T2, C_grid = build_cost_map(env)

env.reset()
controller.reset()
state = env.get_state()

CAPTURE_STEPS = [0, 50, 100, 200, 300, 400]
snapshots = []

def wrap_angle(q):
    return np.arctan2(np.sin(q), np.cos(q))

for t in range(500):
    action, info = controller.plan_step(state)
    obs, cost, done, _ = env.step(action)

    if t in CAPTURE_STEPS:
        rollout_qpos = env.state_qpos(controller._last_states)  # (K, H, 2)
        snapshots.append({
            "t": t,
            "cost_min": info["cost_min"],
            "current_qpos": env.data.qpos.copy(),                # (2,)
            "rollout_shoulder": wrap_angle(rollout_qpos[:, :, 0]),  # (K, H)
            "rollout_elbow": wrap_angle(rollout_qpos[:, :, 1]),
            "weights": controller._last_weights.copy(),
        })

    state = env.get_state()

env.close()

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for i, snap in enumerate(snapshots):
    ax = axes[i]
    ax.pcolormesh(np.degrees(T1), np.degrees(T2), C_grid,
                shading="auto", cmap="viridis", alpha=0.6)

    w = snap["weights"]
    w_norm = w / (w.max() + 1e-12)

    for k in range(w.shape[0]):
        alpha = float(np.clip(w_norm[k] * 5.0, 0.02, 0.8))
        ax.plot(
            np.degrees(snap["rollout_shoulder"][k]),
            np.degrees(snap["rollout_elbow"][k]),
            color="white", alpha=alpha, linewidth=0.3,
        )

    # current position (red star) and goal (green cross at 0,0)
    ax.plot(np.degrees(wrap_angle(snap["current_qpos"][0])),
            np.degrees(wrap_angle(snap["current_qpos"][1])),
            "r*", markersize=12, markeredgecolor="black", markeredgewidth=0.5)
    ax.plot(0, 0, "g+", markersize=15, markeredgewidth=2)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.set_title(f"t={snap['t']}  cost_min={snap['cost_min']:.1f}")
    ax.set_xlabel("Shoulder (deg)")
    ax.set_ylabel("Elbow (deg)")

plt.tight_layout()
plt.savefig("acrobot_rollout_evolution.png", dpi=150)
plt.show()


