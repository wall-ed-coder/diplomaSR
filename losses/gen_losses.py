import torch
from torch import nn as nn, Tensor

from losses.abc_loss import CustomLoss, gen_losses_parameters


class FullyGenLoss(CustomLoss):

    def __init__(self, gen_losses: list = gen_losses_parameters['losses'],
                 gen_weights: list = gen_losses_parameters['weights'],
                 disc_loss: nn.Module = torch.nn.BCEWithLogitsLoss(), disc_loss_weight=2.):
        # here was used realistic discriminator loss for discriminator
        super().__init__()
        assert len(gen_losses) == len(gen_weights)
        self.gen_losses = gen_losses
        self.disc_loss = disc_loss
        self.disc_loss_weight = disc_loss_weight
        gen_weights = torch.tensor(gen_weights)
        self.gen_weights = torch.tensor(gen_weights / torch.sum(gen_weights), requires_grad=True)

    def forward(self, pred_img, real_img, disc_pred_on_fake=None, disc_pred_on_real=None) -> Tensor:
        return self.get_gen_loss(pred_img, real_img) \
               + self.get_disc_loss(disc_pred_on_fake, disc_pred_on_real) \
               * self.disc_loss_weight

    def get_gen_loss(self, pred_img, real_img) -> Tensor:
        return torch.mean(
            torch.tensor([
                loss(pred_img, real_img) for loss in self.gen_losses
            ]) * self.gen_weights
        )

    def get_disc_loss(self, disc_pred_on_fake, disc_pred_on_real) -> Tensor:
        if disc_pred_on_fake is not None and disc_pred_on_real is not None and self.disc_loss is not None:
            return self.disc_loss(disc_pred_on_fake, disc_pred_on_real)
        else:
            return torch.tensor(0., requires_grad=True)