# src/utils/seeding.py
import numpy as np
import torch

def seed_everything(seed: int) -> int:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op without CUDA
    return seed

def add_seed_arg(parser, default: int = 0) -> None:
    parser.add_argument("--seed", type=int, default=default,
                        help="seed for numpy / torch RNGs")