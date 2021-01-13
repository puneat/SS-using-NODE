import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

def count_parameters(model):
    """
    Count number of tunable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)