"""Diagonal Gaussian MLP for GPS"""
# c step refers to controller step 
# s step refers to supervised step 

import numpy as np 
import torch 
import torch.nn as nn 
from src.utils.config import PolicyConfig

class GaussianPolicy(nn.Module):
    """the network produces mu and log std for each action dim"""
    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        
        self.obs_dim = obs_dim 
        self.act_dim = act_dim 

        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        act_fn = activations[cfg.activation]
        
        # MLP; obs_dim -> hidden[0] -> hidden[1] -> 2 * act_dim
        layers = []
        in_dim = obs_dim 
        for h in cfg.hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h

        layers.append(nn.Linear(in_dim, 2 * act_dim))

        self.net = nn.Sequential(*layers)

        # initialise log sigma around 0 so that std starts around 1 
        nn.init.zeros_(self.net[-1].bias[act_dim:])
        self.optimizer = torch.optim.Adam(self.parameters(), lr = cfg.lr)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, obs_dim) → mu: (B, act_dim), log_sigma: (B, act_dim)"""
        out = self.net(obs)
        mu, log_sigma = out[..., :self.act_dim], out[..., self.act_dim: ]
        return mu, log_sigma 
    
    # this is used when GPS compares to the mppi for scoring actions 
    # how likely is the action given my current policy 
    def log_prob(self,
                 obs: torch.Tensor, 
                 actions: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self.forward(obs)
        sigma = log_sigma.exp()
        lp = -0.5 * (((actions - mu) / sigma) ** 2 + 2 * log_sigma + np.log(2 * np.pi))
        return lp.sum(dim=-1)
    
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        # sample actions from π_θ(·|x). Returns (B, act_dim)."""
        mu, log_sigma = self.forward(obs)
        return mu + log_sigma.exp() * torch.randn_like(mu)
    
    # supervised destillation of mppi into the neural network
    def train_weighted(
            self, 
            obs: np.ndarray, 
            actions: np.ndarray, 
            weights: np.ndarray, 
    ) -> float:
        obs_t = torch.as_tensor(obs, dtype = torch.float32)
        act_t = torch.as_tensor(actions, dtype = torch.float32)
        w_t = torch.as_tensor(weights, dtype = torch.float32)

        lp = self.log_prob(obs_t, act_t) # (N, )
        # this is just the negative weight * log likelihood 
        loss = -(w_t * lp).sum() / w_t.sum().clamp(min=1e-8)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    # this is the numpy variant for the log prob function since mppi uses numpy 
    @torch.no_grad()
    def log_prob_np(self,
                    obs: np.ndarray,
                    actions: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(np.array(obs, dtype=np.float32, copy=True))
        act_t = torch.from_numpy(np.array(actions, dtype=np.float32, copy=True))
        return self.log_prob(obs_t, act_t).numpy()
    


        
    


        

        
    

        


        