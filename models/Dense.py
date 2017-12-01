import torch
import torch.nn as nn
import math
import torch.nn.init

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):

    def __init__(self, num_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(num_channels, num_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(num_channels, num_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(num_channels, num_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = conv3x3(num_channels, num_channels)
        self.relu4 = nn.ReLU(inplace=True)


    def forward(self, x):
        a = self.relu1(self.conv1(x))
        b = self.relu2(self.conv2(a))
        c = self.relu3(self.conv3(b))
        d = self.relu4(self.conv4(c))

        a = torch.cat((a, b, c, d), 1)

        return a


class Dense(nn.Module):

    def __init__(self, num_channels=3):
        super(Dense, self).__init__()

        self.start_conv = nn.Sequential(
	    	nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=int(7/2), bias=True),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.layer1 = BasicBlock(64)

        self.transition1 = nn.Sequential(
	    	nn.Conv2d(64*4, 64, kernel_size=1, stride=1, padding=0, bias=True),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.layer2 = BasicBlock(64)

        self.last_conv = nn.Sequential(
	    	nn.Conv2d(64*4, 128, kernel_size=3, stride=1, padding=int(3/2), bias=True),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True),
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
        a = self.layer1(a)
        a = self.transition1(a)
        a = self.layer2(a)
        a = self.last_conv(a)

        return a
