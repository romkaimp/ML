import torch
from numpy.fft import rfft
from torch.fx.experimental.meta_tracer import nn_layernorm_override
from torch.nn import Module
from torch.fft import rfft, irfft
import torch
import torch.nn as nn
import torch.nn.functional as F

class BranchNet(nn.Module):
    """Обрабатывает входные функции (например, u(x))."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)

class TrunkNet(nn.Module):
    """Обрабатывает точки (y), где вычисляется оператор."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y):
        return self.net(y)

class DeepONet1d(nn.Module):
    """Собирает Branch и Trunk сети."""
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dims, output_dim):
        super().__init__()
        self.branch = BranchNet(branch_input_dim, hidden_dims, output_dim)
        self.trunk = TrunkNet(trunk_input_dim, hidden_dims, output_dim)

    def forward(self, u, y):
        # u: [batch_size, branch_input_dim] — входные функции
        # y: [batch_size, trunk_input_dim] — точки предсказания
        branch_out = self.branch(u)  # [batch_size, output_dim]
        trunk_out = self.trunk(y)    # [batch_size, output_dim]
        return torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
