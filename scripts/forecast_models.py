"""Two preregistered cross-channel forecasting architectures."""
import torch
from torch import nn


class CrossChannelForecast(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == 'cross_linear':
            self.net = nn.Linear(480, 120)
        elif kind == 'residual_mlp':
            self.net = nn.Sequential(nn.Linear(480, 128), nn.ReLU(), nn.Linear(128, 120))
        else:
            raise ValueError(kind)
        self.kind = kind

    def forward(self, x):
        return self.net(x.reshape(len(x), 480)).reshape(-1, 8, 15) + x[:, -1:, :]
