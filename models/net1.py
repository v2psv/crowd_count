import torch
import torch.nn as nn
import math
import torch.nn.init

class NET1(nn.Module):

    def __init__(self, num_channels=3, use_pmap=False):
        super(NET1, self).__init__()
        self.use_pmap = use_pmap

        self.img_conv1_1 = nn.Sequential(
	    	nn.Conv2d(num_channels, 32, kernel_size=9, stride=1, padding=int(9/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            )
        self.img_conv1_2 = nn.Sequential(
	    	nn.Conv2d(num_channels, 32, kernel_size=7, stride=1, padding=int(7/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            )
        self.img_conv1_3 = nn.Sequential(
	    	nn.Conv2d(num_channels, 32, kernel_size=5, stride=1, padding=int(5/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            )
        self.img_conv1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.img_conv2 = nn.Sequential(
	    	nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(64),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(64),
	    	nn.ReLU(inplace=True),
	        nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.img_conv3 = nn.Sequential(
	    	nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True),
	    	# nn.ReLU(inplace=True),
            )

        if use_pmap:
            self.pmap_conv1 = nn.Sequential(
    	    	nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=int(3/2), bias=True),
    	    	nn.ReLU(inplace=True),
    	        nn.MaxPool2d(kernel_size=2, stride=2),
                )

            self.pmap_conv2 = nn.Sequential(
    	    	nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(3/2), bias=True),
    	    	nn.ReLU(inplace=True),
    	        nn.MaxPool2d(kernel_size=2, stride=2),
                )

            # self.pmap_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            # self.pmap_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()


    def forward(self, img, pmap=None):
        a = self.img_conv1_1(img)
        b = self.img_conv1_2(img)
        c = self.img_conv1_3(img)
        a = torch.cat((a, b, c), 1)
        a = self.img_conv1_pool(a)

        if self.use_pmap:
            d = self.pmap_conv1(pmap)
            a += d

        a = self.img_conv2(a)

        if self.use_pmap:
            d = self.pmap_conv2(d)
            a += d

        a = self.img_conv3(a)

        return a
