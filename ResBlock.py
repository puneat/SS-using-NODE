import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class ResBlock(nn.Module):
    """
    Simple residual block used to construct ResNet.
    """
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.gn1 = norm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=5, padding=1, bias=False)
        self.gn2 = norm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Shortcut
        identity = x

        # First convolution
        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        # Second convolution
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # Add shortcut
        out += identity

        return out