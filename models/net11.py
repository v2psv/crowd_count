import torch
import torch.nn as nn
import math
import torch.nn.init


class BasicBlock(nn.Module):

    def __init__(self, dim):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )

    def forward(self, x):
        a = self.block(x)

        return a


class SPPBlock(nn.Module):
    def __init__(self, ksize, dim):
        super(SPPBlock, self).__init__()
        self.down_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=ksize, stride=ksize),
            nn.Conv2d(dim, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.1,inplace=False)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        a = self.down_layer(x)
        self.interp = nn.Upsample(size=(h, w), mode='bilinear')
        a = self.interp(a)

        return a


class NET11(nn.Module):

    def __init__(self, in_dim=3):
        super(NET11, self).__init__()

        self.start_conv = nn.Sequential(
	    	nn.Conv2d(in_dim, 32, kernel_size=7, stride=1, padding=3, bias=True),
            # nn.BatchNorm2d(32),
	    	nn.LeakyReLU(negative_slope=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.conv1 = BasicBlock(64)
        self.conv2 = BasicBlock(64)
        self.conv3 = BasicBlock(64)

        self.transition = nn.Sequential(
	    	nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
	    	nn.LeakyReLU(negative_slope=0.1,inplace=False),
            )

        self.spp1 = SPPBlock(2, 64)
        self.spp2 = SPPBlock(4, 64)
        self.spp3 = SPPBlock(6, 64)

        self.last_conv = nn.Sequential(
	    	nn.Conv2d(64 + 16*3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(64),
	    	nn.LeakyReLU(negative_slope=0.1,inplace=False),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, img, pmap=None):
        a = self.start_conv(img)
        b = self.conv1(a)
        c = self.conv2(b)
        d = self.conv3(c)
        a = torch.cat([b, c, d], 1)
        a = self.transition(a)
        x = self.spp1(a)
        y = self.spp2(a)
        z = self.spp3(a)
        a = torch.cat([a, x, y, z], 1)
        a = self.last_conv(a)

        return a
