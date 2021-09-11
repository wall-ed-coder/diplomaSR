from torch import Tensor
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, model: nn.Module):
        super(Generator, self).__init__()
        self.model = model

    def forward(self, batch) -> Tensor:
        return self.model(batch)



