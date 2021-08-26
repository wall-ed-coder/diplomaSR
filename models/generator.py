from torch import nn, Tensor


class Generator(nn.Module):

    def __init__(self, model: nn.Module):
        super(Generator, self).__init__()
        self.model = model

    def forward(self, batch) -> Tensor:
        return self.model(batch)



