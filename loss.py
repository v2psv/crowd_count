import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class ContrastiveLoss2(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss2, self).__init__()

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance-label, 2))

        return loss_contrastive

class MSELoss(_Loss):
    def forward(self, pred, target):
        # _assert_no_grad(target)
        loss = torch.sum((pred - target)**2) / pred.size(0)
        return loss

class RelativeLoss(_Loss):
    def forward(self, pred, target):
        # _assert_no_grad(target)
        loss = torch.sum(((pred-target)/(target+1))**2) / pred.size(0)
        return loss

class LogMSELoss(_Loss):
    def forward(self, pred, target):
        # _assert_no_grad(target)
        loss = torch.sum((torch.log(pred+1) - torch.log(target+1))**2) / pred.size(0)
        return loss

class L1Loss(_Loss):
    def __init__(self, size_average=True, reduce=True, relative=False):
        super(L1Loss, self).__init__(size_average)
        self.reduce = reduce
        self.size_average = size_average
        self.relative = relative

    def forward(self, input, target):
        # _assert_no_grad(target)
        if self.relative:
            input = input / target
            target = target / target
        return F.l1_loss(input, target, size_average=self.size_average)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=[1,3,10,100,1000], size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        weight = torch.from_numpy(np.array(weight)).type(torch.cuda.FloatTensor)
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        if targets.dim() == 4:
            n, c, h, w = targets.size()
            targets = targets.contiguous().view(n, h, w)
        return self.nll_loss((1 - F.softmax(inputs, dim=1)) ** self.gamma * F.log_softmax(inputs, dim=1), targets)

class OrderLoss(_Loss):
    def __init__(self, num=[1,1,1,1]):
        super(OrderLoss, self).__init__()
        n = np.sqrt(num)
        self.weight = torch.from_numpy(n/np.sum(n))

    def forward(self, pred, target):
        n, c, h, w = target.size()
        W = self.weight.clone().repeat(n, h, w, 1).transpose(2, 3).transpose(1, 2).contiguous()
        W = torch.autograd.Variable(W, requires_grad=False).type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
        # cross_entropy = nn.BCELoss(weight=W)
        # loss = cross_entropy(pred, target)
        loss = F.binary_cross_entropy(pred, target, weight=W)
        # loss = torch.log(pred + 1e-10).mul(target) + torch.log(1 - pred + 1e-10).mul((1-target))
        # loss = loss * W
        # loss = -loss.mean()
        return loss
