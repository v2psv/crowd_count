import numpy as np
import torch
from torch import nn
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=1, use_bn=False, activation="ReLU"):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_chan))

        if activation == 'ReLU':
            layers.append(nn.ReLU(inplace=False))
        elif activation == 'LeakyReLU':
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif activation == 'SELU':
            layers.append(nn.SELU(inplace=False))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_chan, out_chan=[32, 32], ksizes=[1, 3], use_bn=False, activation="ReLU"):
        super(InceptionBlock, self).__init__()
        n = len(ksizes)
        layers = [ConvBlock(in_chan, out_chan[i], ksizes[i], stride=1, pad=int(ksizes[i]/2), use_bn=use_bn, activation=activation)] * n
        self.modules = nn.ModuleList(*layers)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.modules):
            outputs.append(layer(x))

        return torch.cat(outputs, 1)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize, stride, use_bn=False, activation="ReLU"):
        super(BasicBlock, self).__init__()
        layers = []
        for i in range(len(in_chan)):
            layers.append(ConvBlock(in_chan[i], out_chan[i], ksize=ksize[i], stride=stride[i], pad=int(ksize[i]/2), use_bn=use_bn, activation=activation))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        a = self.block(x)
        return a


class DenseBlock(nn.Module):
    def __init__(self, dim, use_bn=False):
        super(DenseBlock, self).__init__()
        self.block1 = InceptionBlock(dim, [dim/2, dim/2], ksizes=[1, 3], use_bn=use_bn)
        self.block2 = InceptionBlock(dim, [dim/2, dim/2], ksizes=[1, 3], use_bn=use_bn)
        self.block3 = InceptionBlock(dim, [dim/2, dim/2], ksizes=[1, 3], use_bn=use_bn)

    def forward(self, x):
        a = self.block1(x)
        b = self.block1(a)
        c = self.block1(b)

        return torch.cat([a, b, c], 1)


class DownBlock(nn.Module):
    def __init__(self, ksize, in_chan, out_chan, use_bn=False, pool_type='max', activation="ReLU"):
        super(DownBlock, self).__init__()
        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=ksize, stride=ksize)
        else:
            self.pool = nn.MaxPool2d(kernel_size=ksize, stride=ksize)
        self.conv = ConvBlock(in_chan, out_chan, 3, stride=1, pad=1, use_bn=use_bn, activation=activation)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        self.interp = nn.Upsample(size=(h, w), mode='bilinear')

        a = self.conv(self.pool(x))
        a = self.interp(a)

        return a


class UpBlock(nn.Module):
    def __init__(self, ksize, in_chan, out_chan, use_bn=False, pool_type='max', activation="ReLU"):
        super(UpBlock, self).__init__()
        self.ksize = ksize
        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=ksize, stride=ksize)
        else:
            self.pool = nn.MaxPool2d(kernel_size=ksize, stride=ksize)
        self.conv = ConvBlock(in_chan, out_chan, 3, stride=1, pad=1, use_bn=use_bn, activation=activation)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        h, w = h * self.ksize, w * self.ksize
        self.interp = nn.Upsample(size=(h, w), mode='bilinear')

        a = self.interp(x)
        a = self.pool(self.conv(a))


        return a
