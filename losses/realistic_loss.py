import torch
from torch import nn as nn, Tensor

from losses.abc_loss import CustomLoss


class RealisticLoss(CustomLoss):

    def __init__(self):
        super().__init__()
        self.adversarial_loss: nn.Module = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred_on_fake, pred_on_real) -> Tensor:
        real_loss = self.adversarial_loss(pred_on_real - pred_on_fake, torch.ones_like(pred_on_real))
        fake_loss = self.adversarial_loss(pred_on_real - pred_on_real, torch.zeros_like(pred_on_real))
        d_loss = (real_loss + fake_loss) / 2
        return d_loss
