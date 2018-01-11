import torch
import torch.nn as nn
import math


class MCNN(nn.Module):

    def __init__(self, num_channels=3):
        super(MCNN, self).__init__()
        self.column1 = self._make_column(in_channels=[num_channels, 16, 32, 16],
        								 out_channels=[16, 32, 16, 8],
        								 ksize=[9, 7, 7, 7],
        								 stride=[1, 1, 1, 1])
        self.column2 = self._make_column(in_channels=[num_channels, 20, 40, 20],
        								 out_channels=[20, 40, 20, 10],
        								 ksize=[7, 5, 5, 5],
        								 stride=[1, 1, 1, 1])
        self.column3 = self._make_column(in_channels=[num_channels, 24, 48, 24],
        								 out_channels=[24, 48, 24, 12],
        								 ksize=[5, 3, 3, 3],
        								 stride=[1, 1, 1, 1])
        self.conv_x = nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()


    def _make_column(self, in_channels, out_channels, ksize, stride):
        column = nn.Sequential(
        	nn.Conv2d(in_channels[0], out_channels[0], kernel_size=ksize[0],
        			stride=stride[0], padding=int(ksize[0]/2), bias=True),
        	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels[1], out_channels[1], kernel_size=ksize[1],
        			stride=stride[1], padding=int(ksize[1]/2), bias=True),
        	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels[2], out_channels[2], kernel_size=ksize[2],
        			stride=stride[2], padding=int(ksize[2]/2), bias=True),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(in_channels[3], out_channels[3], kernel_size=ksize[3],
        			stride=stride[3], padding=int(ksize[3]/2), bias=True),
        	nn.ReLU(inplace=True),
        )
        return column


    def forward(self, x, pmap=None):
        a = self.column1(x)
        b = self.column2(x)
        c = self.column3(x)
        d = torch.cat((a, b, c), 1)
        d = self.conv_x(d)

        return d
