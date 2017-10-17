import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class NET3(nn.Module):

    def __init__(self, num_channels=3):
        super(NET3, self).__init__()
        self.num_channels = num_channels

        self.layer1 = self._make_layer(32, 9)
        self.layer2 = self._make_layer(32, 7)
        self.layer3 = self._make_layer(32, 5)
        self.layer4 = self._make_layer(32, 3)
        self.conv_x = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, planes, blocks):
        downsample = nn.Sequential(
                        nn.Conv2d(self.num_channels, planes, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(planes),
                        )

        layers = []
        layers.append(BasicBlock(self.num_channels, planes, stride=1, downsample=downsample))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(BasicBlock(planes, planes, stride=1, downsample=None))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(blocks):
            layers.append(BasicBlock(planes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, pmap=None):
    	a = self.layer1(x)
    	b = self.layer2(x)
    	c = self.layer3(x)
        d = self.layer4(x)
    	d = torch.cat((a, b, c, d), 1)
    	d = self.conv_x(d)

        return d
