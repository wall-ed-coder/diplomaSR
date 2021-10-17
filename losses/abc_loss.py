import torch
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


if __name__ == '__main__':
    from losses.gen_losses import FullyGenLoss

    loss = FullyGenLoss()
    print(torch.sum(loss.gen_weights))
    x = torch.rand((3, 3, 256, 256))
    y = torch.rand((3, 3, 256, 256))

    print(loss(x, y))


