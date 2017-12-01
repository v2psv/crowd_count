import torch
import torch.nn as nn
import math
import torch.nn.init


class DilationBlock(nn.Module):

    def __init__(self, num_channels, dilation=[1,1,1,1], padding=[1,1,1,1]):
        super(DilationBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                               padding=padding[0], dilation=dilation[0], bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                               padding=padding[1], dilation=dilation[1], bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                               padding=padding[2], dilation=dilation[2], bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                               padding=padding[3], dilation=dilation[3], bias=True)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        a = self.relu1(self.conv1(x))
        b = self.relu2(self.conv2(a))
        c = self.relu3(self.conv3(b))
        d = self.relu4(self.conv4(c))

        a = torch.cat((a, b, c, d), 1)

        return a


class DilatedNet(nn.Module):

    def __init__(self, num_channels=3):
        super(DilatedNet, self).__init__()

        self.start_conv = nn.Sequential(
	    	nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=int(7/2), bias=True),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.dilation_layer = DilationBlock(64, [2, 4, 8, 16], [2, 4, 8, 16])

        self.last_conv = nn.Sequential(
            nn.Conv2d(64*4, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)
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
        a = self.dilation_layer(a)
        a = self.last_conv(a)

        return a
