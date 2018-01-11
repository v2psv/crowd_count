import torch
import torch.nn as nn
import torch.nn.init
from torchvision import models
from models.common_blocks import ConvBlock, BasicBlock, FeaturePyramid


class Siamese_Perspect(nn.Module):
    def __init__(self, in_dim=3, use_bn=False, activation="ReLU"):
        super(Siamese_Perspect, self).__init__()

        self.feature = nn.Sequential(
            ConvBlock(in_dim, 32, ksize=7, stride=1, pad=3, use_bn=use_bn, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(32, 64, ksize=5, stride=1, pad=2, use_bn=use_bn, activation=activation),
            FeaturePyramid(in_dim=64, use_bn=use_bn, activation=activation),
            ConvBlock(192, 128, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 1/16
            )

        self.fc = nn.Sequential(
            nn.Linear(8*8*128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
        )

    def forward_once(self, x):
        output = self.feature(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
