import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid


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
            ConvBlock(128, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            )

        self.last_layer = nn.Sequential(
            ConvBlock(128, 1024, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
            ConvBlock(1024, out_chan, ksize=1, stride=1, pad=0, use_bn=False, activation=activation)
            )

    def forward(self, feature1, feature2, feature3):
        x = self.up1(feature3)
        x = self.decoder1(torch.cat([feature2, x], 1))
        x = self.up2(x)
        x = self.decoder2(torch.cat([feature1, x], 1))
        out = self.last_layer(x)

        return out, x

class Scene_Embed2(nn.Module):
    def __init__(self, in_dim=3, use_bn=True, activation="ReLU", n_class=5, pretrained=False):
        super(Scene_Embed2, self).__init__()
        self.pretrained = pretrained
        # 1/2
        self.conv1 = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=9, stride=2, pad=4, use_bn=use_bn, activation=activation),
            ConvBlock(32, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
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

        self.context  = Decoder(5, use_bn=use_bn, activation=activation)
        self.perspect = Decoder(1, use_bn=use_bn, activation=activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        y = self.conv3(x)
        z = self.conv4(y)

        context, f1  = self.context(x, y, z)
        perspect, f2 = self.perspect(x, y, z)

        if self.pretrained:
            return context, perspect, f1, f2
        else:
            return context, perspect
