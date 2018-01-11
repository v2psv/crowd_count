import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import DownBlock, ConvBlock, BasicBlock, UpBlock


class ResNet50_Skip(nn.Module):
    def __init__(self, activation="ReLU"):
        super(ResNet50_Skip, self).__init__()
        pretrained_model = models.resnet50(pretrained=True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.layer1 = nn.Sequential(*(list(pretrained_model.children())[:6]))
        self.layer2 = pretrained_model.layer3
        self.layer3 = pretrained_model.layer4
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False

        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            )
        self.decoder2 = nn.Sequential(
            ConvBlock(3072, 1024, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            nn.Upsample(scale_factor=2),
            )
        self.decoder3 = nn.Sequential(
            ConvBlock(1536, 128, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 1, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            )

        self.init_weight(self.decoder1.modules())
        self.init_weight(self.decoder2.modules())
        self.init_weight(self.decoder3.modules())

    def init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)

        out = self.decoder1(feature3)
        out = self.decoder2(torch.cat([feature2, out], 1))
        out = self.decoder3(torch.cat([feature1, out], 1))

        return out
