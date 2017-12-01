import torch
import torch.nn as nn
import math
import torch.nn.init


class BasicBlock(nn.Module):

    def __init__(self, dim):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        a = self.relu1(self.conv1(x))
        a = self.relu2(self.conv2(a))
        a = self.relu3(self.conv3(a))

        return a


class SPPBlock(nn.Module):
    def __init__(self, ksize, dim):
        super(SPPBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=ksize, stride=ksize)
        # self.pool = nn.AvgPool2d(kernel_size=ksize, stride=ksize)
        self.conv = nn.Conv2d(dim, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        a = self.pool(x)
        a = self.conv(a)
        a = self.relu(a)

        h, w = x.size(2), x.size(3)

        self.interp = nn.Upsample(size=(h, w), mode='bilinear')
        a = self.interp(a)

        return a


class Dense_SPP(nn.Module):

    def __init__(self, in_dim=3):
        super(Dense_SPP, self).__init__()

        self.start_conv = nn.Sequential(
	    	nn.Conv2d(in_dim, 32, kernel_size=7, stride=1, padding=3, bias=True),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.conv1 = BasicBlock(64)
        self.conv2 = BasicBlock(64)
        self.conv3 = BasicBlock(64)

        self.transition = nn.Sequential(
	    	nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=True),
	    	nn.ReLU(inplace=True),
            )

        self.spp1 = SPPBlock(2, 64)
        self.spp2 = SPPBlock(4, 64)
        self.spp3 = SPPBlock(6, 64)

        self.last_conv = nn.Sequential(
	    	nn.Conv2d(64 + 16*3, 64, kernel_size=3, stride=1, padding=1, bias=True),
	    	nn.ReLU(inplace=True),
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
