import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint


def norm(dim):
    """
    Group normalization to improve model accuracy and training speed.
    """
    return nn.GroupNorm(min(32, dim), dim)
    