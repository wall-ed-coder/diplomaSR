import torch
from torch import nn


class DiscriminatorTest(nn.Module):
    def __init__(self):
        super(DiscriminatorTest, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.ones(x.shape[0]).resize(x.shape[0], 1)