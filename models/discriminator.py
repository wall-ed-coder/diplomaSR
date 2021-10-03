from typing import Tuple

from torch import nn, Tensor


class Discriminator(nn.Module):

    def __init__(self, model: nn.Module):
        super(Discriminator, self).__init__()
        self.model = model

    def forward(self, pred_batch_sr_img, real_batch_sr_img) -> Tuple[Tensor, Tensor]:
        return self.model(pred_batch_sr_img), self.model(real_batch_sr_img)


class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # for input 256/256
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [128, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [256, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [512, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)

        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(512 * 8 * 8, 512)
        self.linear2 = nn.Linear(512, 1)
        # activation function

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = {
            (16, 16): nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            (32, 24): nn.AvgPool2d(kernel_size=(5, 4), stride=(4, 3), padding=1),
            (32, 32): nn.AvgPool2d(kernel_size=5, stride=4, padding=1),
            (64, 48): nn.Sequential(
                self.lrelu, self.conv4_0, self.bn4_0, self.lrelu,
                self.conv4_1, self.bn4_1,
                nn.AvgPool2d(kernel_size=(5, 4), stride=(4, 3), padding=1),
            ),
            (64, 64): nn.Sequential(
                self.lrelu, self.conv4_0, self.bn4_0, self.lrelu,
                self.conv4_1, self.bn4_1,
                nn.AvgPool2d(kernel_size=5, stride=4, padding=1),
            ),
        }

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.bn3_1(self.conv3_1(fea))

        fea = self.avg_pool[fea.shape[-2:]](fea)

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)

        return self.sigmoid(out)


if __name__ == '__main__':
    import torch
    from preprocessing.preprocessing import SIZES_FOR_CROPS

    d = Discriminator_VGG_128(3, 64)
    for size in SIZES_FOR_CROPS:
        print(d(torch.rand((1, 3, *size))))
