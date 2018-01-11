import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid


class Scene_Embed(nn.Module):
    def __init__(self, in_dim=3, use_bn=True, activation="ReLU", n_class=5):
        super(Scene_Embed, self).__init__()

        # 1/4
        self.feature = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=11, stride=1, pad=5, use_bn=use_bn, activation=activation),
            ConvBlock(32, 64, ksize=5, stride=1, pad=2, use_bn=use_bn, activation=activation),
            FeaturePyramid(in_dim=64, use_bn=use_bn, activation=activation),
            ConvBlock(64*3, 256, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            )

        self.context = nn.Sequential(
            ConvBlock(256, 1024, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
            ConvBlock(1024, n_class, ksize=1, stride=1, pad=0, use_bn=False, activation=activation)
            )

        self.perspect = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4),
            ConvBlock(256, 1024, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
            ConvBlock(1024, 1, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img):
        x = self.feature(img)
        context = self.context(x)
        perspect = self.perspect(x)

        return context, perspect
