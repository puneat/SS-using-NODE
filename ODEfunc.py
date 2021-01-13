import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class ODEfunc(nn.Module):
    """
    Network architecture for ODENet.
    """
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0    # Number of function evaluations

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out
