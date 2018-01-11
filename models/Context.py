import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid


class Context(nn.Module):
    def __init__(self, in_dim=3, use_bn=True, activation="ReLU", n_class=5):
        super(Context, self).__init__()

        self.feature = nn.Sequential(
                ConvBlock(in_dim, 32, ksize=11, stride=1, pad=5, use_bn=use_bn, activation=activation),
                ConvBlock(32, 64, ksize=5, stride=1, pad=2, use_bn=use_bn, activation=activation),
                FeaturePyramid(in_dim=64, use_bn=use_bn, activation=activation),
                ConvBlock(64*3, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            )

        self.classifier = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvBlock(128, 1024, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
                nn.Dropout2d(),
                ConvBlock(1024, 512, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
                nn.Dropout2d(),
                ConvBlock(512, n_class, ksize=1, stride=1, pad=0, use_bn=False, activation=activation),
                nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2)
                # nn.Upsample(scale_factor=2)
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
        context = self.classifier(x)

        return context
