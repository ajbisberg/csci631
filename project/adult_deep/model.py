import torch
import torch.nn as nn


class AdultModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(13, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.sequence(x)

    def name(self):
        return "AdultModel"
