import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import DownBlock, ConvBlock, BasicBlock, UpBlock


class VGG_Skip(nn.Module):
    def __init__(self, activation="ReLU"):
        super(VGG_Skip, self).__init__()
        pretrained_model = models.vgg16(pretrained=True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.layer1 =  nn.Sequential(*list(pretrained_model.features.children())[:16])
        self.layer2 =  nn.Sequential(*list(pretrained_model.features.children())[16:23])
        self.layer3 =  nn.Sequential(*list(pretrained_model.features.children())[23:30])
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
            ConvBlock(1024, 256, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            nn.Upsample(scale_factor=2),
            )
        self.decoder3 = nn.Sequential(
            ConvBlock(512, 128, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(128, 1, ksize=3, stride=1, pad=1, use_bn=False, activation=activation)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.init_weight(self.decoder1.modules())
        self.init_weight(self.decoder2.modules())
        self.init_weight(self.decoder3.modules())

        '''
        self.feature = nn.Sequential(*list(pretrained_model.features.children())[:30])
        for param in self.feature.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(512, 256, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 128, ksize=3, stride=1, pad=1, use_bn=False, activation=activation),
            ConvBlock(128, 1, ksize=3, stride=1, pad=1, use_bn=False, activation=activation)
            )
        self.init_weight(self.decoder.modules())
        '''
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
        # print feature1.size(), feature2.size(), feature3.size()
        out = self.decoder1(feature3)
        out = self.decoder2(torch.cat([feature2, out], 1))
        out = self.decoder3(torch.cat([feature1, out], 1))

        # out = self.decoder(self.feature(x))
        return out
