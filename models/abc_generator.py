from torch import Tensor
import torch
import torch.nn as nn

from models.generator_test import Generator_Test


class Generator(nn.Module):

    def __init__(self, model: nn.Module):
        super(Generator, self).__init__()
        self.model = model

    def forward(self, batch) -> Tensor:
        return torch.clamp(self.model(batch), min=0., max=1.)


if  __name__ == '__main__':
    # import torchsummary
    network = Generator(model=Generator_Test(4))
    t = torch.rand((10, 3, 256, 256))
    print(network(t).shape)

    # network = Generator(model=RRDBNet(3,3,64,4,8,64))
    # print(network(t).shape)
