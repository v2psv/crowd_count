import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid

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


class Perspective(nn.Module):
    def __init__(self, in_dim=3, use_bn=True, activation="ReLU"):
        super(Perspective, self).__init__()
        """
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        self.feature1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        """

        """
        pretrained_model = models.vgg16(pretrained=True)
        for param in pretrained_model.parameters():
            param.requires_grad = False
        self.encoder = pretrained_model.features
        self.encoder_fc = nn.Linear(8*8*512, 10)
        self.decoder_fc = nn.Linear(10, 8*8*128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )
        self.init_weight(self.decoder.modules())
        """

        self.encoder = Encoder(in_dim, use_bn=use_bn, activation=activation)
        self.decoder = Decoder(out_chan=64, use_bn=use_bn, activation=activation)

        self.perspect = nn.Sequential(
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


    def init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img):
        """
        feature = self.encoder(img)
        # feature = self.spatial_pyramid_pooling(feature, (8, 8))
        feature = feature.view(feature.size(0), -1)
        embed = self.encoder_fc(feature)
        feature = self.decoder_fc(embed)
        feature = feature.view(feature.size(0), 128, 8, 8)
        out = self.decoder(feature)
        """
        f1, f2, f3 = self.encoder(img)
        f_p = self.decoder(f1, f2, f3)
        out = self.perspect(f_p)

        return out, f_p
