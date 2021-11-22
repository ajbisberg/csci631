import torch
import torch.nn as nn


class AdultModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(91, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequence(x)

    def name(self):
        return "AdultModel"
