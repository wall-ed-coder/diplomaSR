from math import log2

from torch import nn as nn


class Generator_Test(nn.Module):
    def __init__(self, n_super_resolution=4):
        super(Generator_Test, self).__init__()
        assert n_super_resolution in [2, 4, 8, 16]
        self.linear = nn.Linear(1, 1)
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),) for _ in range(int(log2(n_super_resolution)))
        ])

    def forward(self, x):
        return self.layers(x)