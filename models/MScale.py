import torch
import torch.nn as nn
import math
import torch.nn.init
from models.common_blocks import DownBlock, ConvBlock, BasicBlock, UpBlock


class MScale(nn.Module):

    def __init__(self, in_dim=3, use_bn=False, activation="ReLU"):
        super(MScale, self).__init__()

        self.start_conv = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=7, stride=1, pad=3, use_bn=use_bn, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.conv1 = BasicBlock(in_chan=[64, 64, 64], out_chan=[64, 64, 64], ksize=[3, 3, 3], stride=[1, 1, 1], use_bn=use_bn, activation=activation)
        self.conv2 = BasicBlock(in_chan=[64, 64, 64], out_chan=[64, 64, 64], ksize=[3, 3, 3], stride=[1, 1, 1], use_bn=use_bn, activation=activation)
        self.conv3 = BasicBlock(in_chan=[64, 64, 64], out_chan=[64, 64, 64], ksize=[3, 3, 3], stride=[1, 1, 1], use_bn=use_bn, activation=activation)

        self.transition = ConvBlock(64*3, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation)

        self.spp1 = UpBlock(2, in_chan=64, out_chan=64, use_bn=use_bn, pool_type='max', activation=activation)
        self.spp2 = DownBlock(2, in_chan=64, out_chan=64, use_bn=use_bn, pool_type='max', activation=activation)

        self.last_conv = nn.Sequential(
            ConvBlock(64 + 64 * 2, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, img):
        a = self.start_conv(img)
        b = self.conv1(a)
        c = self.conv2(b)
        d = self.conv3(c)
        a = torch.cat([b, c, d], 1)
        a = self.transition(a)
        x = self.spp1(a)
        y = self.spp2(a)
        a = torch.cat([a, x, y], 1)
        a = self.last_conv(a)

        return a
