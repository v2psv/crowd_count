import torch
import torch.nn as nn
import math
import torch.nn.init


class ReceptiveCount(nn.Module):

    def __init__(self, in_dim=3, receptive=64):
        super(ReceptiveCount, self).__init__()
        self.redundant = receptive * receptive

        self.start_conv = nn.Sequential(
	    	nn.Conv2d(in_dim, 16, kernel_size=7, stride=1, padding=receptive, bias=True),
	    	nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=0, bias=True),
	    	nn.LeakyReLU(negative_slope=0.1,inplace=False),
            )

        self.block1 = nn.Sequential(
        nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1,inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1,inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1,inplace=False),
            )

        self.block2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1,inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1,inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1,inplace=False),
            )

        self.last_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
	    	nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, bias=True),
	    	nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()


    def forward(self, img):

        a = self.start_conv(img)
        b = self.block1(a)
        c = self.block2(b)
        d = self.last_conv(c) / self.redundant

        return d
