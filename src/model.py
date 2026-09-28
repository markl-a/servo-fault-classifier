import torch
from torch import nn


class Conv1DNet(nn.Module):
    """Small 1D CNN: strided convs act as learnable filter banks; global pooling makes it length-agnostic."""

    def __init__(self, n_classes: int = 5, in_ch: int = 1):
        super().__init__()
        def block(i, o, k, s):
            return nn.Sequential(nn.Conv1d(i, o, k, s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.net = nn.Sequential(block(in_ch, 16, 64, 4), block(16, 32, 16, 2), block(32, 64, 8, 2), block(64, 64, 8, 2),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, x):  # x: (B, 1, N)
        return self.net(x)
