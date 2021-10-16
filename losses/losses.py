import torch
from torch import Tensor
import torch.nn as nn
from piq import FSIMLoss, PieAPP, DISTS, GMSDLoss, MultiScaleGMSDLoss, MDSILoss, HaarPSILoss

gen_losses_parameters = {
    'weights': [2., 2., 2., 2.5, 3., 2., 2., 2., 2., 5., 3.,],
    'losses': [
        HaarPSILoss(),
        MDSILoss(),
        GMSDLoss(),
        MultiScaleGMSDLoss(chromatic=True),
        DISTS(mean=[0., 0., 0.], std=[1., 1., 1.]),

        PieAPP(stride=27, enable_grad=True),
        PieAPP(stride=36, enable_grad=True),
        PieAPP(stride=16, enable_grad=True),

        FSIMLoss(chromatic=True),

        nn.MSELoss(),
        nn.L1Loss(),
    ],
}


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.get_loss(*args, **kwargs)


class BasicGenLoss(CustomLoss):

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

    def get_loss(self, pred_img, real_img, disc_pred_on_fake=None, disc_pred_on_real=None) -> Tensor:
        return self.get_gen_loss(pred_img, real_img) \
               + self.get_disc_loss(disc_pred_on_fake, disc_pred_on_real) \
               * self.disc_loss_weight

    def get_gen_loss(self, pred_img, real_img) -> Tensor:
        return torch.mean(
            torch.tensor([
                loss(pred_img.clone().detach(), real_img.clone().detach()) for loss in self.gen_losses
            ]) * self.gen_weights
        )

    def get_disc_loss(self, disc_pred_on_fake, disc_pred_on_real) -> Tensor:
        if disc_pred_on_fake is not None and disc_pred_on_real is not None and self.disc_loss is not None:
            return self.disc_loss(disc_pred_on_fake - disc_pred_on_real, torch.full_like(disc_pred_on_fake, 1.))
        else:
            return torch.tensor(0., requires_grad=True)


class RealisticDiscLoss(CustomLoss):

    def __init__(self):
        super().__init__()
        self.disc_loss: nn.Module = torch.nn.BCEWithLogitsLoss()

    def get_loss(self, disc_pred_on_fake, disc_pred_on_real) -> Tensor:
        return self.disc_loss(disc_pred_on_real - disc_pred_on_fake, torch.full_like(disc_pred_on_fake, 1.))


if __name__ == '__main__':
    loss = BasicGenLoss()
    print(torch.sum(loss.gen_weights))
    x = torch.rand((3, 3, 256, 256))
    y = torch.rand((3, 3, 256, 256))

    print(loss.get_loss(x, y))


