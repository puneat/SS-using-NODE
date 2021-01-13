import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class Flatten(nn.Module):
    """
    Flatten feature maps for input to linear layer.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)