"""dataclass configurations for mppi-gps"""
import json
from dataclasses import dataclass
from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

@dataclass 
class MPPIConfig:
    K: int = 256 # number of samples 
    H: int = 50 # planning horizon 
    lam: float = 1.0 # temperature parameter 
    # this parameter essentially helps you know how much you want to focus on specific samples vs others 
    noise_sigma: float = 0.5 # exploration noise std 
    adaptive_lam: bool = False # adapt lam in order to maintain the n_eff
    n_eff_threshold: float = 64.0 # number of samples that you want to contribute to the weighted mean

    @staticmethod
    def load(env_name: str) -> "MPPIConfig":
        """Load best tuned params from configs/<env_name>_best.json."""
        path = _CONFIGS_DIR / f"{env_name}_best.json"
        params = json.loads(path.read_text())
        return MPPIConfig(**params)

@dataclass
class PolicyConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    activation: str = "relu"

@dataclass
class GPSConfig:
    num_iterations: int = 50
    num_conditions: int = 5         # number of initial states
    episode_length: int = 500       # steps per episode during GPS training
    kl_estimator: str = "moment_matched" # "moment_matched" (Eq 3): fits Gaussian to MPPI samples, closed-form KL — stable but unimodal
    # "sample_based" (Eq 4): estimates KL directly from weighted particles variance
    kl_target: float = 1.0          # target KL for BADMM dual update
    badmm_init_nu: float = 1.0     # initial dual variable (penalizes KL between MPPI and policy)
    badmm_step_size: float = 2.0
    policy_augmented_alpha: float = 0.1 # weight on -log π(u|x) in MPPI cost (Eq 5)
    distill_batch_size: int = 256   # mini-batch size for policy distillation
    distill_epochs: int = 5         # gradient epochs per GPS iteration
    warm_start_policy: bool = True  # warm-start MPPI nominal U from policy rollout

