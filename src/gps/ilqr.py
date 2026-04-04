"""
The implementation of iLQR for the baseline GPS 
- Linearise the mujoco dynamics through the use of mjd transition 
through finite differencing and the run the ricatti equation to produce time 
varying linear gaussian controllers for GPS destillation  
"""

import numpy as np 
import mujoco 
from dataclasses import dataclass 

from src.envs.mujoco_env import MuJoCoEnv

@dataclass 
class iLQRConfig:
    max_iters: int = 50 
    tol: float = 1e-6  # convergence tolerance on cost improvement
    fd_eps: float = 1e-6 # finite-difference epsilon for cost derivatives
    mu_min: float =  1e-6 # minimum regularization
    mu_max: float = 1e10 # maximum regularization
    mu_init: float  = 1.0 # initial regularisation 
    delta_0: float = 2.0 # regularisation scaling 
    alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)  # line search steps

class iLQR:
    pass 
