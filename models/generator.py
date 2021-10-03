from torch import Tensor
import functools
import torch
import torch.nn as nn
from math import log2


class Generator(nn.Module):

    def __init__(self, model: nn.Module):
        super(Generator, self).__init__()
        self.model = model

    def forward(self, batch) -> Tensor:
        return self.model(batch)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, n_blocks, n_super_resolution=4, hidden_dim_in_blocks=32):
        super(RRDBNet, self).__init__()
        assert n_super_resolution in [2, 4, 8, 16]
        RRDB_block_f = functools.partial(RRDB, nf=hidden_dim, gc=hidden_dim_in_blocks)

        self.conv_first = nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, n_blocks)
        self.trunk_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(hidden_dim, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #### upsampling

        layers = []
        for _ in range(int(log2(n_super_resolution))):
            layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
                    self.lrelu,
                )
            )
        self.upconv = nn.Sequential(*layers)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.upconv(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


if  __name__ == '__main__':
    # import torchsummary
    t = torch.rand((1, 3, 256, 256))
    network = Generator(model=RRDBNet(3,3,64,4,8,64))
    print(network(t).shape)
