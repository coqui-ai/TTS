import torch
from torch import nn
from torch.nn import functional as F


class Postnet(nn.Module):
    def __init__(self, channels=128, kernel_size=33, n_layers=12):
        super().__init__()
        assert n_layers >= 2, " [!] n_layers must be greater than or equal to 2"
        assert kernel_size % 2 == 1, " [!] kernel_size must be an odd number"
        layers = [
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Tanh(),
        ]
        for _ in range(n_layers - 2):
            layers.append(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.Tanh())
        layers.append(nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=kernel_size // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
