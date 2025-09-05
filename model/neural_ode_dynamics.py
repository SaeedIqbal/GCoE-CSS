"""
Implements Neural ODE-based Latent-Output Co-Evolution.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class TimeGatedResidualBlock(nn.Module):
    """Time-gated block with positional encoding."""
    def __init__(self, D, C, D_prime=64, D_double=32):
        super().__init__()
        self.conv_z = nn.Conv2d(D, D_prime, kernel_size=3, padding=1)
        self.conv_y = nn.Conv2d(C, D_prime, kernel_size=3, padding=1)
        self.linear_t = nn.Linear(D_prime, D_double)
        self.bias = nn.Parameter(torch.zeros(1, D_prime, 1, 1))
        self.bias_t = nn.Parameter(torch.zeros(1, D_double, 1, 1))
        self.swish = nn.SiLU()
        self.T_stage = 50

    def forward(self, h_t, t):
        z_t, y_t = h_t[:, :-y_t.shape[1]], h_t[:, -y_t.shape[1]:]
        pe = torch.sin(2 * 3.14159 / self.T_stage * t)
        pe = pe.view(1, -1, 1, 1).expand(z_t.size(0), -1, z_t.size(2), z_t.size(3))

        out = torch.relu(self.conv_z(z_t) + self.conv_y(y_t) + self.bias)
        out = self.linear_t(out.permute(0,2,3,1)).permute(0,3,1,2) + self.bias_t
        return self.swish(out)


class NeuralODECoEvolution(nn.Module):
    """Neural ODE for joint evolution of features and logits."""
    def __init__(self, D, C):
        super().__init__()
        self.F_psi = TimeGatedResidualBlock(D, C)
        self.tol_abs = 1e-5
        self.tol_rel = 1e-4

    def forward(self, h0, t_span):
        solution = odeint(
            self.F_psi, h0, t_span.to(h0.device),
            method='dopri5',
            atol=self.tol_abs,
            rtol=self.tol_rel
        )
        return solution[-1]