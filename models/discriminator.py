from typing import Tuple

from torch import nn, Tensor


class Discriminator(nn.Module):

    def __init__(self, model: nn.Module):
        super(Discriminator, self).__init__()
        self.model = model

    def forward(self, pred_batch_sr_img, real_batch_sr_img) -> Tuple[Tensor, Tensor]:
        return self.model(pred_batch_sr_img), self.model(real_batch_sr_img)

