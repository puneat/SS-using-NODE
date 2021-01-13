import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint

class ODENet(nn.Module):
    """
    Neural ODE.
    Uses ODE solver (dopri5 by default) to yield model output.
    Backpropagation is done with the adjoint method as described in
    https://arxiv.org/abs/1806.07366.
    Parameters
    ----------
    odefunc : nn.Module
        network architecture
    rtol : float
        relative tolerance of ODE solver
    atol : float
        absolute tolerance of ODE solver
    """
    def __init__(self, odefunc, rtol=1e-3, atol=1e-3):
        super(ODENet, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint_adjoint(self.odefunc, x, self.integration_time, self.rtol, self.atol)
        return out[1]

    # Update number of function evaluations (nfe)
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value