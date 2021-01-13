import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class ConcatConv1d(nn.Module):
    """
    1d convolution concatenated with time for usage in ODENet.
    """
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=True, transpose=False):
        super(ConcatConv1d, self).__init__()
        module = nn.ConvTranspose1d if transpose else nn.Conv1d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
