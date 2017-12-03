import torch
import torch.nn as nn
import math
import torch.nn.init
from models.common_blocks import DownBlock, ConvBlock, BasicBlock, UpBlock


class ContexModule(nn.Module):
    def __init__(self, in_dim, n_class, use_bn=False, activation="ReLU"):
        super(ContexModule, self).__init__()

        self.start_block = BasicBlock(in_chan=[in_dim, 32], out_chan=[32, 64], ksize=[7, 5], stride=[1, 1], use_bn=use_bn, activation=activation)
        self.classifier = nn.Sequential(
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            nn.Dropout2d(),
            ConvBlock(64, 64, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation),
            nn.Dropout2d(),
            ConvBlock(64, n_class, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation)
        )

    def forward(self, fmap):
        a = self.start_block(fmap)
        a = self.classifier(a)

        return a


class MScale_Cascade(nn.Module):
    def __init__(self, in_dim=3, use_bn=False, activation="ReLU", use_contex=True, n_class=5):
        super(MScale_Cascade, self).__init__()
        self.use_contex = use_contex

        self.start_conv = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=7, stride=1, pad=3, use_bn=use_bn, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation)
            )

        self.block1 = nn.Sequential(BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 64], ksize=[3, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation))
        self.block2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 64], ksize=[3, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation))
        self.block3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 128], ksize=[3, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation),
                                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_contex:
            self.contex_block = ContexModule(64*3, n_class, use_bn=use_bn, activation=activation)
            last_conv_in = 64 * 3 + n_class
        else:
            last_conv_in = 64 * 3

        self.last_conv = nn.Sequential(
            ConvBlock(last_conv_in, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
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
        x = self.block1(a)
        y = self.block2(x)
        z = self.block3(y)
        x = self.pool(x)

        b = torch.cat([x, y, z], 1)

        if self.use_contex:
            c = self.contex_block(b)
            b = torch.cat([b, c], 1)
            a = self.last_conv(b)
            return a, c
        else:
            a = self.last_conv(b)
            return a


class MScale_Cascade_Rmap(nn.Module):
    def __init__(self, in_dim=3, use_bn=False, activation="ReLU"):
        super(MScale_Cascade_Rmap, self).__init__()

        self.start_conv = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=7, stride=1, pad=3, use_bn=use_bn, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation)
            )

        self.block1 = nn.Sequential(BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 64], ksize=[3, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation))
        self.block2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 64], ksize=[3, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation))
        self.block3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 128], ksize=[3, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation),
                                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.last_conv = nn.Sequential(
            ConvBlock(64 * 3 + 1, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img, rmap):
        a = self.start_conv(img)
        b = self.block1(a)
        x = self.block2(b)
        y = self.block3(x)
        z = self.pool(b)
        b = torch.cat([x, y, z, rmap], 1)
        a = self.last_conv(b)

        return a
