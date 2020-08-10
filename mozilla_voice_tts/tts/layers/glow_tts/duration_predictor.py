import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, dropout_p):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        self.drop = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(in_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_1 = nn.GroupNorm(1, filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.norm_2 = nn.GroupNorm(1, filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
