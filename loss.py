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
    def __init__(self):
        super(MSELoss, self).__init__()

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


class PmapLoss(_Loss):
    def __init__(self, ksize=15):
        self.ksize = ksize
        self.avg_pool = nn.AvgPool1d(kernel_size=ksize, stride=ksize)

    def forward(self, pred, target, avg_density, mask):
        x = self.avg_pool(torch.sum(pred, dim=3)) * self.ksize
        y = self.avg_pool(torch.sum(target, dim=3)) * self.ksize

        n1 = y[:, :, :-1]
        n2 = y[:, :, 1:]

class GradientLoss(_Loss):
    def __init__(self, alpha=1):
        super(GradientLoss, self).__init__()
        self.alpha = alpha
        self.pad_left = nn.ConstantPad2d((1,0,0,0), 0)
        self.pad_top  = nn.ConstantPad2d((0,0,1,0), 0)

    def forward(self, pred, true):
        x1 = torch.abs(pred[:,:,:,1:] - pred[:,:,:,:-1])
        x2 = torch.abs(true[:,:,:,1:] - true[:,:,:,:-1])
        y1 = torch.abs(pred[:,:,1:,:] - pred[:,:,:-1,:])
        y2 = torch.abs(true[:,:,1:,:] - true[:,:,:-1,:])

        x1 = self.pad_left(x1)
        x2 = self.pad_left(x2)
        y1 = self.pad_top(y1)
        y2 = self.pad_top(y2)

        loss = torch.sum(torch.abs(x1-x2)**self.alpha+torch.abs(y1-y2)**self.alpha) / pred.size(0)

        return loss

class L2_Grad_Loss(_Loss):
    def __init__(self, alpha=1, lambda_g=1):
        super(L2_Grad_Loss, self).__init__()
        self.lambda_g = lambda_g
        self.alpha = alpha
        self.pad_left = nn.ConstantPad2d((1,0,0,0), 0)
        self.pad_top  = nn.ConstantPad2d((0,0,1,0), 0)

    def forward(self, pred, true):
        l2_loss = torch.sum((pred - true)**2) / pred.size(0)

        x1 = torch.abs(pred[:,:,:,1:] - pred[:,:,:,:-1])
        x2 = torch.abs(true[:,:,:,1:] - true[:,:,:,:-1])
        y1 = torch.abs(pred[:,:,1:,:] - pred[:,:,:-1,:])
        y2 = torch.abs(true[:,:,1:,:] - true[:,:,:-1,:])

        x1 = self.pad_left(x1)
        x2 = self.pad_left(x2)
        y1 = self.pad_top(y1)
        y2 = self.pad_top(y2)

        grad_loss = torch.sum((x1-x2)**self.alpha + (y1-y2)**self.alpha) / pred.size(0)

        return l2_loss + self.lambda_g * grad_loss

class KLLoss(_Loss):
    def forward(self, pred, target):
        loss = torch.sum(target*(torch.log(target+1e-6) - torch.log(pred+1e-6))) / pred.size(0)
        return loss

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
        inputs += 1e-6
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
