import torch
import torch.nn as nn
import math
import torch.nn.init
from models.common_blocks import DownBlock, ConvBlock, BasicBlock, UpBlock


class ContexModule(nn.Module):
    def __init__(self, in_dim, n_class, use_bn=False, activation="ReLU"):
        super(ContexModule, self).__init__()

        self.down_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), #1/8
            ConvBlock(in_dim, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            # nn.AvgPool2d(kernel_size=2, stride=2), #1/16
            # ConvBlock(128, 256, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
        )
        self.classifier = nn.Sequential(
            ConvBlock(128, 512, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation),
            nn.Dropout2d(),
            ConvBlock(512, 512, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation),
            nn.Dropout2d(),
            ConvBlock(512, n_class, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation)
        )
        self.up_block = nn.Sequential(nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2))

    def forward(self, fmap):
        a = self.down_block(fmap)
        a = self.classifier(a)
        a = self.up_block(a)

        return a


class ReceptiveModule(nn.Module):
    def __init__(self, in_dim, use_bn=False, activation="ReLU"):
        super(ReceptiveModule, self).__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/8
            ConvBlock(in_dim, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1/16
            ConvBlock(128, 256, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
        )

    def forward(self, fmap):
        a = self.block(fmap)
        h, w = a.size(2), a.size(3)
        nn.Conv2d(256, 256, kernel_size=[h, w], stride=1, padding=0)

        return a


class MScale_Cascade(nn.Module):
    def __init__(self, in_dim=3, use_bn=False, activation="ReLU", use_contex=True, n_class=5):
        super(MScale_Cascade, self).__init__()
        self.use_contex = use_contex

        self.start_conv = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=7, stride=1, pad=3, use_bn=use_bn, activation=activation),
            ConvBlock(32, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation)
            )

        self.block1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 64], ksize=[3, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation))
        self.block2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 64], ksize=[5, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation))
        self.block3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    BasicBlock(in_chan=[64, 64, 64, 64], out_chan=[64, 64, 64, 128], ksize=[7, 3, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn, activation=activation),
                                    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_contex:
            self.contex_block = ContexModule(64*3, n_class, use_bn=use_bn, activation=activation)
            last_conv_in = 64 * 3 + n_class
        else:
            last_conv_in = 64 * 3

        use_bn = False
        self.last_conv = nn.Sequential(
            ConvBlock(last_conv_in, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
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
        c = self.contex_block(b)

        if self.use_contex:
            b = torch.cat([b, c], 1)
            a = self.last_conv(b)
        else:
            a = self.last_conv(b)

        return a, c
