from abc import ABC
import torch.nn as nn
from typing import List


class LessBlock(nn.Module, ABC):
    def __init__(self, in_channels: int, create_channels: int, out_channels: int, activation: nn.Module = nn.ReLU):
        super(LessBlock, self).__init__()
        assert create_channels >= out_channels

        self.out_channels = out_channels
        self.create_channels = create_channels

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, create_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(create_channels),
            activation(),
            nn.Conv2d(create_channels, create_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(create_channels),
            activation(),
        )

    def forward(self, x):
        x = self.main(x)
        if self.create_channels == self.out_channels:
            return x
        x = x[:, :self.out_channels]
        return x


class NetworkLessLayers(nn.Module, ABC):

    def __init__(
            self,
            n_classes: int = 10,
            n_blocks: int = 3,
            shapes_net: List[List] = ([3, 64, 32], [32, 128, 64], [64, 128, 128]),
            default_activation: nn.Module = nn.Hardswish
    ):
        super(NetworkLessLayers, self).__init__()

        assert n_blocks == len(shapes_net)
        self.shapes_net = shapes_net
        self.default_activation = default_activation

        self.init_activations()

        self.blocks = nn.Sequential(
            *[
                LessBlock(in_channels=t[0], create_channels=t[1], out_channels=t[2], activation=t[3])
                for t in self.shapes_net
            ],
        )
        self.net = nn.Sequential(

            nn.Conv2d(in_channels=self.shapes_net[-1][-2], out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            self.default_activation(),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            self.default_activation(),

            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            self.default_activation(),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=3072, out_features=n_classes)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.net(x)
        x = x.view(-1, 3072)
        x = self.linear(x)
        return x

    def init_activations(self):
        new_shapes_net = []
        for t in self.shapes_net:
            l = len(t)
            if l == 4:
                new_shapes_net.append(t)
            elif l == 3:
                new_shapes_net.append(t + [self.default_activation])
            else:
                raise NotImplemented
        self.shapes_net = new_shapes_net
