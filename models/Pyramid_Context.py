import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid


class ContextModule(nn.Module):
    def __init__(self, in_dim, n_class, down_scale=2, use_bn=False, activation="ReLU"):
        super(ContextModule, self).__init__()
        module_list = []
        for i in range(down_scale//2):
            module_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
            module_list.append(BasicBlock(in_chan=[in_dim, 128, 128], out_chan=[128, 128, 128], ksize=[3, 3, 3], stride=[1, 1, 1], use_bn=use_bn, activation=activation))
            in_dim = 128

        self.feature = nn.Sequential(*module_list)
        self.classifier = nn.Sequential(
            # ConvBlock(128, 512, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation),
            ConvBlock(128, n_class, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation),
            # nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2),
            nn.Upsample(scale_factor=down_scale, mode='bilinear')
        )

    def forward(self, fmap):
        a = self.feature(fmap)
        a = self.classifier(a)

        return a


class Pyramid_Context(nn.Module):
    def __init__(self, in_dim=3, use_bn=True, activation="ReLU", n_class=1):
        super(Pyramid_Context, self).__init__()

        self.start_conv = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=11, stride=1, pad=5, use_bn=use_bn, activation=activation),
            ConvBlock(32, 64, ksize=7, stride=1, pad=3, use_bn=use_bn, activation=activation),
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation)
            )

        self.feature = FeaturePyramid(in_dim=64, use_bn=use_bn, activation=activation)
        self.context1 = ContextModule(192, n_class=n_class, down_scale=2, use_bn=False, activation=activation)
        self.context2 = ContextModule(192, n_class=n_class, down_scale=4, use_bn=False, activation=activation)

        self.density = nn.Sequential(
            ConvBlock(192, 128, ksize=5, stride=1, pad=2, use_bn=False, activation=activation),
            ConvBlock(128, 64, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(64, 64, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(64, 1, ksize=1, stride=1, pad=0, use_bn=False, activation=activation)
            )

    def forward(self, img):
        x = self.start_conv(img)
        x = self.feature(x)

        context1 = self.context1(x)
        context2 = self.context2(x)
        # density = self.density(torch.cat([x, context1, context2], 1))
        density = self.density(x)

        return density, context1, context2
