import torch
import torch.nn as nn
import math
import torch.nn.init

class NET2(nn.Module):

    def __init__(self, num_channels=3, use_pmap=False):
        super(NET2, self).__init__()
        self.use_pmap = use_pmap
        self.img_conv1_1 = nn.Sequential(
	    	nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.img_conv1_2 = nn.Sequential(
	    	nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.img_conv1_3 = nn.Sequential(
	    	nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.img_conv1_4 = nn.Sequential(
	    	nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(32),
	    	nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.img_conv2 = nn.Sequential(
	    	nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(64),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(64),
	    	nn.ReLU(inplace=True),
	        nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.img_conv3 = nn.Sequential(
	    	nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(3/2), bias=True),
            nn.BatchNorm2d(64),
	    	nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
            )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                m.bias.data.zero_()


    def forward(self, img, pmap=None):
        a = self.img_conv1_1(img)
        b = self.img_conv1_2(img)
        c = self.img_conv1_3(img)
        d = self.img_conv1_4(img)
        a = torch.cat((a, b, c, d), 1)

        a = self.img_conv2(a)

        a = self.img_conv3(a)

        return a
