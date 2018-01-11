import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_blocks import ConvBlock, BasicBlock

class MCNN_Attention(nn.Module):

    def __init__(self, in_dim=3, use_bn=False, activation='ReLU'):
        super(MCNN_Attention, self).__init__()
        self.column1 = BasicBlock(in_chan=[in_dim, 16, 32, 32, 32, 32], out_chan=[16, 32, 32, 32, 32, 64],
                                  ksize=[11, 9, 7, 7, 5, 3, 3], stride=[2, 1, 2, 1, 1, 1], use_bn=use_bn, activation=activation)
        self.column2 = BasicBlock(in_chan=[in_dim, 32, 32, 32, 32, 64], out_chan=[32, 32, 32, 32, 64, 64],
                                  ksize=[9, 7, 5, 5, 3, 3], stride=[2, 1, 2, 1, 1, 1], use_bn=use_bn, activation=activation)
        self.column3 = BasicBlock(in_chan=[in_dim, 32, 32, 64, 64, 64], out_chan=[32, 32, 64, 64, 64, 64],
                                  ksize=[7, 5, 3, 3, 3, 3], stride=[2, 1, 2, 1, 1, 1], use_bn=use_bn, activation=activation)

        self.attention = nn.Sequential(
            ConvBlock(192, 512, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(512, 3, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation),
        )

        self.last_layer = nn.Sequential(
            ConvBlock(192, 64, ksize=3, stride=1, pad=1, use_bn=use_bn, activation=activation),
            ConvBlock(64, 1, ksize=1, stride=1, pad=0, use_bn=use_bn, activation=activation),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        a = self.column1(x)
        b = self.column2(x)
        c = self.column3(x)

        w = F.softmax(self.attention(torch.cat([a,b,c], 1)), dim=1)
        a = a * w[:,0:1,:,:].repeat(1, 64, 1, 1).contiguous()
        b = b * w[:,1:2,:,:].repeat(1, 64, 1, 1).contiguous()
        c = c * w[:,2:,:,:].repeat(1, 64, 1, 1).contiguous()

        density = self.last_layer(torch.cat([a, b, c], 1))
        return density, None
