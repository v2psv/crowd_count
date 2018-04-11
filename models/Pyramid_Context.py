import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid


class ContextModule(nn.Module):
    def __init__(self, in_dim, n_class, down_scale, use_bn=False, activation="ReLU"):
        super(ContextModule, self).__init__()

        self.feature = nn.Sequential(
            ConvBlock(in_dim, 128, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(128, 128, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            nn.AvgPool2d(kernel_size=down_scale, stride=1, padding=(int(down_scale//2), int(down_scale//2))),
        )

        self.classifier = nn.Sequential(
            ConvBlock(128, 128, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(128, n_class, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
            # nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2),
        )

    def forward(self, fmap):
        a = self.feature(fmap)
        a = self.classifier(a[:,:,:-1,:-1])

        return a


class Pyramid_Context(nn.Module):
    def __init__(self, in_dim=3, use_bn=True, activation="ReLU", n_class=5, use_pmap=False):
        super(Pyramid_Context, self).__init__()
        self.use_pmap = use_pmap

        self.start_conv = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=11, stride=1, pad=5, use_bn=use_bn, activation=activation),
            ConvBlock(32, 32, ksize=5, stride=1, pad=2, use_bn=use_bn, activation=activation),
            )

        in_dim = 32
        if use_pmap: in_dim += 1

        # 1/4
        self.feature = FeaturePyramid(in_dim=in_dim, use_bn=use_bn, activation=activation)

        # 1/64
        self.context = ContextModule(224, n_class=n_class, down_scale=16, use_bn=False, activation=activation)

        self.density = nn.Sequential(
            ConvBlock(224, 128, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(128, 64, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
            ConvBlock(64, 1, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
            )

    def forward(self, img, pmap=None):
        # if self.use_pmap and pmap is not None:
            # img = torch.cat([img, pmap], 1)

        x = self.start_conv(img)

        if self.use_pmap and pmap is not None:
            x = torch.cat([x, pmap], 1)

        x = self.feature(x)
        context = self.context(x)

        # x = torch.cat([context, x], 1)

        # if self.use_pmap and pmap is not None:
            # x = torch.cat([x, pmap[:, ::4, ::4]], 1)

        density = self.density(x)

        return density, context
