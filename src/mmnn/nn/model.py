# SPDX-FileCopyrightText: 2026-present zach wick <zach@btrbet.app>
#
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn


class BracketPredictor(nn.Module):
    """MLP for binary classification: predicts if higher-ranked team wins (1) or lower-ranked (0)."""

    def __init__(self, input_size: int = 9, hidden_sizes: tuple[int, ...] = (64, 32, 16)):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), nn.ReLU()])
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)
