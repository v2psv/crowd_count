import torch
import torch.nn as nn
import math
import torch.nn.init
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid
from models.Pyramid_Context import Pyramid_Context
from models.Perspective import Perspective


class Encoder(nn.Module):
    def __init__(self, in_chan, use_bn=True, activation="ReLU"):
        super(Encoder, self).__init__()

        # 1/2
        self.conv1 = nn.Sequential(
            ConvBlock(in_chan, 32, ksize=11, stride=2, pad=5, use_bn=use_bn, activation=activation),
            ConvBlock(32, 64, ksize=7, stride=1, pad=3, use_bn=use_bn, activation=activation),
            ConvBlock(64, 64, ksize=5, stride=1, pad=2, use_bn=use_bn, activation=activation),
            )

        # 1/4
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(in_chan=[64, 64, 64], out_chan=[64, 64, 64], ksize=[3, 3, 3], stride=[1, 1, 1], use_bn=use_bn, activation=activation),
            )

        # 1/8
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(in_chan=[64, 128, 128], out_chan=[128, 128, 128], ksize=[3, 3, 3], stride=[1, 1, 1], use_bn=use_bn, activation=activation),
            )

        # 1/16
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(in_chan=[128, 128, 128], out_chan=[128, 128, 128], ksize=[3, 3, 3], stride=[1, 1, 1], use_bn=use_bn, activation=activation),
            )

    def forward(self, img):
        x = self.conv1(img)
        f1 = self.conv2(x)
        f2 = self.conv3(f1)
        f3 = self.conv4(f2)

        return f1, f2, f3


class Decoder(nn.Module):
    def __init__(self, out_chan, use_bn=True, activation="ReLU"):
        super(Decoder, self).__init__()

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # 1/8
        self.decoder1 = nn.Sequential(
            ConvBlock(256, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(128, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            )

        # 1/4
        self.decoder2 = nn.Sequential(
            ConvBlock(192, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(128, out_chan, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            )

    def forward(self, feature1, feature2, feature3):
        x = torch.cat([self.up1(feature3), feature2], 1)
        x = self.decoder1(x)
        x = torch.cat([self.up2(x), feature1], 1)
        x = self.decoder2(x)

        return x


class MScale_Cascade(nn.Module):
    def __init__(self, in_dim=3, use_bn=False, activation="ReLU"):
        super(MScale_Cascade, self).__init__()

        self.encoder = Encoder(in_dim, use_bn=use_bn, activation=activation)
        self.density_decoder = Decoder(out_chan=64, use_bn=use_bn, activation=activation)
        self.context_decoder = Decoder(out_chan=64, use_bn=use_bn, activation=activation)

        self.context  = nn.Sequential(
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(64, 5, ksize=1, stride=1, pad=0, use_bn=False, activation=activation)
            )

        self.density = nn.Sequential(
            ConvBlock(64+5, 64, ksize=5, stride=1, pad=2, use_bn=False, activation=activation),
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(64, 1, ksize=1, stride=1, pad=0, use_bn=False, activation=activation)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img, perspect=None):
        f1, f2, f3 = self.encoder(img)
        density = self.density_decoder(f1, f2, f3)
        context = self.context_decoder(f1, f2, f3)
        context = self.context(context)
        density = self.density(torch.cat([density, context], 1))

        return density, context
