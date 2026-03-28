"""Vanilla SL sanity check"""

import numpy as np 
import h5py 
import torch
from src.policy.gaussian_policy import GaussianPolicy
from src.utils.config import PolicyConfig
from pathlib import Path
from src.envs.acrobot import Acrobot
import mediapy 

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco 

demo_path = Path("data/acrobot_demos.h5")
obs_dim = 4 
act_dim = 1 
num_epochs = 50 
horizon_step = 0 # only take t = 0 from each H horizon 

def load_demos(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_obs, all_act, all_w = [], [], []
    
    with h5py.File(path, "r") as f:
        for cond_key in sorted(f.keys()):
            cond = f[cond_key]
            for step_key in sorted(cond.keys()):
                step = cond[step_key]
                # take single horizon index to avoid 200M pairs
                obs = step["obs"][:, horizon_step, :]       # (K, obs_dim)
                actions = step["actions"][:, horizon_step, :]  # (K, act_dim)
                weights = step["weights"][:]                    # (K,)

                all_obs.append(obs)
                all_act.append(actions)
                all_w.append(weights)

    obs = np.concatenate(all_obs, axis=0)        # (N, obs_dim)
    actions = np.concatenate(all_act, axis=0)     # (N, act_dim)
    weights = np.concatenate(all_w, axis=0)       # (N,)

    # renormalise weights so they sum to 1 across entire dataset
    weights /= weights.sum()
    return obs, actions, weights

def main():
    obs, actions, weights = load_demos(str(demo_path))
    print(f"loaded {obs.shape[0]} samples")

    policy = GaussianPolicy(obs_dim, act_dim, PolicyConfig())

    for epoch in range(num_epochs):
        loss = policy.train_weighted(obs, actions, weights)
        print(f"epoch {epoch:3d}  loss={loss:.4f}")

    # roll out the trained policy
    env = Acrobot()
    env.reset()
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    frames = []

    for t in range(500):
        obs_t = torch.as_tensor(env._get_obs(), dtype=torch.float32).unsqueeze(0)
        action = policy.sample(obs_t).squeeze(0).detach().numpy()
        obs, cost, done, _ = env.step(action)

        renderer.update_scene(env.data)
        frames.append(renderer.render().copy())

    mediapy.write_video("policy_sl.mp4", frames, fps=30)
    print("Saved policy_sl.mp4")
    env.close()

if __name__ == "__main__":
    main()
    
