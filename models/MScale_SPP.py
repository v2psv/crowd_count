import torch
import torch.nn as nn
import math
import torch.nn.init
from models.common_blocks import DownBlock, ConvBlock


class MScale_SPP(nn.Module):

    def __init__(self, in_dim=3, use_bn=False):
        super(MScale_SPP, self).__init__()

        self.scale1 = self.make_column(in_chan=[in_dim, 16, 16, 32], out_chan=[16, 16, 32, 16], ksize=[9, 9, 7, 7], stride=[1, 1, 1, 1], use_bn=use_bn)
        self.scale2 = self.make_column(in_chan=[in_dim, 16, 32, 32], out_chan=[16, 32, 32, 16], ksize=[7, 7, 5, 5], stride=[1, 1, 1, 1], use_bn=use_bn)
        self.scale3 = self.make_column(in_chan=[in_dim, 32, 32, 64], out_chan=[32, 32, 64, 32], ksize=[5, 5, 3, 3], stride=[1, 1, 1, 1], use_bn=use_bn)

        self.spp1 = DownBlock(2, 64, 32, use_bn)
        self.spp2 = DownBlock(4, 64, 32, use_bn)
        self.spp3 = DownBlock(8, 64, 32, use_bn)

        self.last_conv = nn.Sequential(
            ConvBlock(64 + 32 * 3, 128, ksize=3, stride=1, pad=1, use_bn=use_bn),
            ConvBlock(128, 64, ksize=1, stride=1, pad=0, use_bn=use_bn),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def make_column(self, in_chan, out_chan, ksize, stride, use_bn):
    	return nn.Sequential(
	    	ConvBlock(in_chan[0], out_chan[0], ksize=ksize[0], stride=stride[0], pad=int(ksize[0]/2), use_bn=use_bn),
	        nn.MaxPool2d(kernel_size=2, stride=2),
	        ConvBlock(in_chan[1], out_chan[1], ksize=ksize[1], stride=stride[1], pad=int(ksize[1]/2), use_bn=use_bn),
	        nn.MaxPool2d(kernel_size=2, stride=2),
	        ConvBlock(in_chan[2], out_chan[2], ksize=ksize[2], stride=stride[2], pad=int(ksize[2]/2), use_bn=use_bn),
            ConvBlock(in_chan[3], out_chan[3], ksize=ksize[3], stride=stride[3], pad=int(ksize[3]/2), use_bn=use_bn),
        )

    def forward(self, img):
        a = self.scale1(img)
        b = self.scale2(img)
        c = self.scale3(img)
        d = torch.cat([a, b, c], 1)
        x = self.spp1(d)
        y = self.spp2(d)
        z = self.spp3(d)
        k = torch.cat([d, x, y, z], 1)
        m = self.last_conv(k)

        return m
