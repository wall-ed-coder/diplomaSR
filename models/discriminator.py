from typing import Tuple
import torch
from torch import nn, Tensor
from math import log2
from preprocessing.preprocessing import MAX_IMG_WIDTH, MAX_IMG_HEIGHT

class Discriminator(nn.Module):

    def __init__(self, model: nn.Module):
        super(Discriminator, self).__init__()
        self.model = model

    def forward(self, pred_batch_sr_img, real_batch_sr_img) -> Tuple[Tensor, Tensor]:
        return self.model(pred_batch_sr_img), self.model(real_batch_sr_img)


class DiscriminatorUsualBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1_0 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.bn1_0 = nn.BatchNorm2d(nf, affine=True)
        self.conv1_1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(nf, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.bn1_0(self.conv1_0(x)))
        x = self.lrelu(self.bn1_1(self.conv1_1(x)))
        return x


class DiscriminatorUsualBlockReducesX2(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1_0 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.bn1_0 = nn.BatchNorm2d(nf, affine=True)
        self.conv1_1 = nn.Conv2d(nf, nf, 4, 2, 1)
        self.bn1_1 = nn.BatchNorm2d(nf, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.bn1_0(self.conv1_0(x)))
        x = self.lrelu(self.bn1_1(self.conv1_1(x)))
        return x


class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.usual_blocks1 = nn.Sequential(*[DiscriminatorUsualBlock(nf) for _ in range(3)])
        max_num_blocks = int(log2(max(MAX_IMG_WIDTH, MAX_IMG_HEIGHT)))-int(log2(32))
        reduce2_usual_block = [
            DiscriminatorUsualBlockReducesX2(nf)
            for _ in range(max_num_blocks)
        ]
        self.reduce2_usual_block1 = {
            i: nn.Sequential(*reduce2_usual_block[:i]) for i in range(max_num_blocks)
        }
        self.usual_blocks2 = nn.Sequential(*[DiscriminatorUsualBlock(nf) for _ in range(3)])

        self.linear1 = nn.Linear(nf * 8 * 8, 512)
        self.linear2 = nn.Linear(512, 1)
        # activation function

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
        # 1:1, 1:2, 1:4
        self.avg_pool = {
            (32, 16): nn.AvgPool2d(kernel_size=(5, 3), stride=(4, 2), padding=1),
            (32, 8): nn.AvgPool2d(kernel_size=(5, 1), stride=(4, 1), padding=(1, 0)),
            (32, 32): nn.AvgPool2d(kernel_size=5, stride=4, padding=1),
        }

    def forward(self, x):
        assert 2**int(log2(x.shape[-1])) == x.shape[-1], x.shape
        assert 2**int(log2(x.shape[-2])) == x.shape[-2], x.shape
        # asserts that height and width are power of 2
        assert max(x.shape[-2:])//min(x.shape[-2:]) in [4, 2, 1]
        # aspect ratio in [4, 2, 1]

        power_of_2_in_max_axis = int(log2(max(x.shape[-1], x.shape[-2])))-int(log2(32))

        x = self.lrelu(self.conv0_0(x))
        x = self.usual_blocks1(x)
        x = self.reduce2_usual_block1[power_of_2_in_max_axis](x)
        x = self.usual_blocks2(x)

        x = self.avg_pool[x.shape[-2:]](x)

        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear1(x))
        out = self.linear2(x)

        return self.sigmoid(out)


class Discriminator_Test(nn.Module):
    def __init__(self):
        super(Discriminator_Test, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.ones(x.shape[0]).resize(x.shape[0], 1)


if __name__ == '__main__':
    from preprocessing.preprocessing import SIZES_FOR_CROPS

    d = Discriminator_VGG_128(3, 8)
    for size in SIZES_FOR_CROPS:
        print(size)
        print(d(torch.rand((1, 3, *size))).shape)
