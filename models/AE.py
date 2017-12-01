import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


class DenseBlock(nn.Module):

    def __init__(self, dim):
        super(BasicBlock, self).__init__()

        self.layer1 = self.get_layer(dim)
        self.layer2 = self.get_layer(dim)
        self.layer3 = self.get_layer(dim)
        self.layer4 = self.get_layer(dim)

    def get_layer(self, dim):
        return nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )

    def forward(self, x):
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        a = torch.cat((a, b, c, d), 1)

        return a


class AE(nn.Module):
    def __init__(self, in_dim=3):
        super(AE, self).__init__()
        self.scale1 = self.make_column(in_channels=[in_dim, 16, 16, 32], out_channels=[16, 16, 32, 16], ksize=[9, 9, 7, 7], stride=[1, 1, 1, 1])
        self.scale2 = self.make_column(in_channels=[in_dim, 16, 32, 32], out_channels=[16, 32, 32, 16], ksize=[7, 7, 5, 5], stride=[1, 1, 1, 1])
        self.scale3 = self.make_column(in_channels=[in_dim, 32, 32, 64], out_channels=[32, 32, 64, 32], ksize=[5, 5, 3, 3], stride=[1, 1, 1, 1])
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(128, 1, 3, stride=1, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0),
            nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
			)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_column(self, in_channels, out_channels, ksize, stride):
    	return nn.Sequential(
	    	nn.Conv2d(in_channels[0], out_channels[0], kernel_size=ksize[0],
	    			stride=stride[0], padding=int(ksize[0]/2), bias=True),
            nn.BatchNorm2d(out_channels[0]),
	    	nn.LeakyReLU(negative_slope=0.1, inplace=False),
	        nn.MaxPool2d(kernel_size=2, stride=2),
	        nn.Conv2d(in_channels[1], out_channels[1], kernel_size=ksize[1],
	    			stride=stride[1], padding=int(ksize[1]/2), bias=True),
            nn.BatchNorm2d(out_channels[1]),
	    	nn.LeakyReLU(negative_slope=0.1, inplace=False),
	        nn.MaxPool2d(kernel_size=2, stride=2),
	        nn.Conv2d(in_channels[2], out_channels[2], kernel_size=ksize[2],
	    			stride=stride[2], padding=int(ksize[2]/2), bias=True),
            nn.BatchNorm2d(out_channels[2]),
	    	nn.LeakyReLU(negative_slope=0.1, inplace=False),
	    	nn.Conv2d(in_channels[3], out_channels[3], kernel_size=ksize[3],
	    			stride=stride[3], padding=int(ksize[3]/2), bias=True),
            nn.BatchNorm2d(out_channels[3]),
	    	nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )

    def forward(self, x):
        a = self.scale1(x)
        b = self.scale2(x)
        c = self.scale3(x)
        d = torch.cat([a, b, c], 1)
        e = self.encoder(d)
        f = self.decoder(e)
        return f
