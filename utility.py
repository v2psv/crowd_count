# -*- coding:utf-8 -*-
import h5py, os
import json
import time
import torch
import numpy as np
from torch.nn.modules.loss import _Loss


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


def print_info(epoch=None, train_time=None, test_time=None, \
                loss=None, error_mae=None, error_mse=None, lr=None):
    """
    epoch: tuple (epoch index, total number of epochs)
    train_time, test_time, loss, error_mae, error_mse, error_rmse: AverageMeter objects
    """
    str = ''
    if epoch is not None:
        str = str + 'Epoch: [{epoch[0]}/{epoch[1]}]   '.format(epoch=epoch)
    if train_time is not None:
        str = str + 'TrainTime {train_time.sum:.3f}   '.format(train_time=train_time)
    if test_time is not None:
        str = str + 'TestTime {test_time.sum:.3f}   '.format(test_time=test_time)
    if loss is not None:
        str = str + 'Loss {loss.avg:.5f}   '.format(loss=loss)
    if error_mae is not None and error_mse is not None:
        rmse = np.sqrt(error_mse.avg)
        str = str + 'Error [{error_mae.avg:.3f} {error_mse.avg:.3f} {rmse:.3f}]   '\
                    .format(error_mae=error_mae, error_mse=error_mse, rmse=rmse)
    if lr is not None:
        str = str + 'Learnig Rate {lr:.3E}'.format(lr=lr)

    if len(str) > 0:
        print(str)


def _check_params(params):
    pass

def load_params(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    _check_params(data)

    return data

def save_args(chkp_dir, args):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    with open(chkp_dir+'/parameters.json', 'w') as fp:
        json.dump(args, fp)


def save_checkpoint(chkp_dir, stats):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    chkp_file = chkp_dir + '/checkpoint.tar'
    torch.save(stats, chkp_file)


def save_pred_result(chkp_dir, train_loss, test_loss, pred_dmap, pred_idx, sample=0):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    result_file = chkp_dir + '/result.h5'
    num_pred = len(pred_dmap)

    with h5py.File(result_file, 'w') as hdf:
        hdf.create_dataset("train_loss", data=train_loss)
        hdf.create_dataset("test_loss", data=test_loss)

        if sample == 0:
            idx = np.arange(num_pred)
        else:
            idx = sorted(np.random.permutation(num_pred)[:sample])

        hdf.create_dataset('img_index', data=[pred_idx[i] for i in idx])
	for i in idx:
            hdf.create_dataset("pred_dmap/"+str(pred_idx[i]), data=pred_dmap[i])


class MSELoss(_Loss):
    def forward(self, input, target):
        # _assert_no_grad(target)
        loss = torch.sum((input - target)**2) / input.size(0)
        return loss


def adjust_learning_rate(optimizer, epoch, base_lr, period=50):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // period))
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
