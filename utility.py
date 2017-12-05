# -*- coding:utf-8 -*-
import h5py, os, sys
import json
import time
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch import nn
from shutil import copyfile


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()


def print_info(epoch=None, train_time=None, test_time=None, \
                dmap_loss=None, contex_loss=None, error_mae=None, \
                error_mse=None, lr=None):
    """
    epoch: tuple (epoch index, total number of epochs)
    train_time, test_time, dmap_loss, error_mae, error_mse, error_rmse: AverageMeter objects
    """
    str = ''
    if epoch is not None:
        str = str + 'Epoch: [{epoch[0]}/{epoch[1]}]   '.format(epoch=epoch)
    if train_time is not None:
        str = str + 'TrainTime {train_time.sum:.3f}   '.format(train_time=train_time)
    if test_time is not None:
        str = str + 'TestTime {test_time.sum:.3f}   '.format(test_time=test_time)
    if dmap_loss is not None:
        str = str + 'Density Loss {dmap_loss.avg:.5f}   '.format(dmap_loss=dmap_loss)
    if contex_loss is not None:
        str = str + 'Contex Loss {contex_loss.avg:.5f}   '.format(contex_loss=contex_loss)
    if error_mae is not None and error_mse is not None:
        rmse = np.sqrt(error_mse.avg)
        str = str + 'Error [{error_mae.avg:.3f} {error_mse.avg:.3f} {rmse:.3f}]   '\
                    .format(error_mae=error_mae, error_mse=error_mse, rmse=rmse)
    if lr is not None:
        str = str + 'Learnig Rate {lr:.3E}'.format(lr=lr)

    if len(str) > 0:
        print(str)


def load_params(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)

    return data


def save_args(chkp_dir, args):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    with open(chkp_dir+'/parameters.json', 'w') as fp:
        json.dump(args, fp)

    model = args['model']['arch'] + '.py'
    copyfile('models/' + model, chkp_dir + '/'  + model)


def save_checkpoint(chkp_dir, stats, mode='newest'):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    if mode == 'newest':
        chkp_file = chkp_dir + '/newest_checkpoint.tar'
    elif mode == 'best':
        chkp_file = chkp_dir + '/best_checkpoint.tar'

    torch.save(stats, chkp_file)


def save_pred_result(chkp_dir, train_loss, test_loss, pred_dmap, pred_contex, pred_idx, sample=0, mode='newest'):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    if mode == 'newest':
        result_file = chkp_dir + '/newest_result.h5'
    elif mode == 'best':
        result_file = chkp_dir + '/best_result.h5'

    num_pred = len(pred_dmap)

    with h5py.File(result_file, 'w') as hdf:
        hdf.create_dataset("train_loss", data=train_loss)
        hdf.create_dataset("test_loss", data=test_loss)

        if sample == 0:
            idx = np.arange(num_pred)
        else:
            idx = sorted(np.random.permutation(num_pred)[:sample])

        hdf.create_dataset('img_index', data=[pred_idx[i] for i in idx])
        hdf.create_dataset('pred_cnt', data=[np.sum(pred_dmap[i]) for i in idx])
        for i in idx:
            hdf.create_dataset("pred_dmap/"+str(pred_idx[i]), data=pred_dmap[i])
            hdf.create_dataset("pred_contex/"+str(pred_idx[i]), data=pred_contex[i])


class MSELoss(_Loss):
    def forward(self, pred, target):
        # _assert_no_grad(target)
        loss = torch.sum((pred - target)**2) / pred.size(0)
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class ContexFocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(ContexFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


class DmapFocalLoss2d(nn.Module):
    def __init__(self, gamma=2, epsilon=0.1, num_patch=[8, 8]):
        super(DmapFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_patch = num_patch

    def forward(self, inputs, targets):
        l2loss = (inputs - targets)**2
        # loss = (pred - target)**2 / pred.size(0) is wrong !!!
        # do not use average loss over batch, because patches in each sample have
        # different loss, so should use different weights

        ksize = (inputs.size(2) / self.num_patch[0], inputs.size(3) / self.num_patch[1])

        pool_x = nn.AvgPool2d(kernel_size=ksize, stride=ksize)
        patch_loss = pool_x(l2loss) * ksize[0] * ksize[1]
        patch_targets = pool_x(targets) * ksize[0] * ksize[1]

        pool_pred = nn.AvgPool2d(kernel_size=ksize, stride=ksize)
        patch_pred = pool_pred(inputs) * ksize[0] * ksize[1]

        patch_weight = (torch.abs(patch_pred - patch_targets) / (patch_targets + self.epsilon)) ** self.gamma

        patch_weight = patch_weight / torch.sum(patch_weight).data[0]
        patch_weight = torch.autograd.Variable(patch_weight.data, requires_grad=False)


        weight = l2loss / ((targets + self.epsilon)**2) + targets
        weight = torch.autograd.Variable(weight.data, requires_grad=False)

        return torch.sum(weight * l2loss)


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class DmapContexLoss(_Loss):
    def __init__(self, weight=None, border=0):
        super(DmapContexLoss, self).__init__()
        # self.weight = torch.from_numpy(weight).type(torch.cuda.FloatTensor)
        self.contex_criterion = ContexFocalLoss2d()
        self.dmap_criterion = MSELoss()
        self.border = border

    def forward(self, pred_dmap, pred_contex, target_dmap, target_contex):
        if self.border != 0:
            pred_contex = pred_contex[:,:,self.border:-self.border,self.border:-self.border]
            target_contex = target_contex[:,:,self.border:-self.border,self.border:-self.border]
        b, n, h, w = target_contex.size()
        target_contex = target_contex.contiguous().view(b, h, w)

        dmap_loss = self.dmap_criterion(pred_dmap, target_dmap)
        contex_loss = self.contex_criterion(pred_contex, target_contex)
        dmap_contex_loss = dmap_loss + 10*contex_loss

        return dmap_loss, contex_loss, dmap_contex_loss


class L1Loss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(L1Loss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        # _assert_no_grad(target)
        return F.l1_loss(input, target, size_average=self.size_average)


def adjust_learning_rate(optimizer, epoch, base_lr, rate=0.1, period=30):
    """Sets the learning rate to the initial LR decayed by rate every period epochs"""
    lr = base_lr * (rate ** (epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class Timer(object):
    """Timer class."""
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        self.interval = self.end - self.start
