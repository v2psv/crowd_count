# -*- coding:utf-8 -*-
import h5py, os, sys
import json
import time
import torch
import numpy as np
import pandas as pd
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
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()


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

def print_info(recorder, epoch=None, preffix='', suffix=''):
    str = preffix
    if epoch is not None:
        str = str + 'Epoch: {epoch}   '.format(epoch=epoch)
    if 'time' in recorder:
        str = str + 'Time {time.sum:.3f}   '.format(time=recorder['time'])
    if 'density_loss' in recorder:
        str = str + 'Density Loss {density_loss.avg:.5f}   '.format(density_loss=recorder['density_loss'])
    if 'context_loss' in recorder:
        str = str + 'Context Loss {context_loss.avg:.5f}   '.format(context_loss=recorder['context_loss'])
    if 'context1_loss' in recorder:
        str = str + 'Context Loss [{context1_loss.avg:.5f} {context2_loss.avg:.5f}]   '.format(context1_loss=recorder['context1_loss'], context2_loss=recorder['context2_loss'])
    if 'perspect_loss' in recorder:
        str = str + 'Perspect Loss {perspect_loss.avg:.5f}   '.format(perspect_loss=recorder['perspect_loss'])
    if 'error_mae' in recorder and 'error_mse' in recorder:
        rmse = np.sqrt(recorder['error_mse'].avg)
        str = str + 'Error [{error_mae.avg:.3f} {mse:.3f}]   '.format(error_mae=recorder['error_mae'], mse=rmse)
    str  = str + suffix
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

    chkp_file = chkp_dir + '/' + mode + '_checkpoint.tar'
    torch.save(stats, chkp_file)


def save_loss(chkp_dir, train_loss, test_loss):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    with pd.HDFStore(chkp_dir + '/loss.h5', 'w') as hdf:
        hdf['train_loss'] = train_loss
        hdf['test_loss'] = test_loss

def save_result(chkp_dir, result_dict, mode='newest', num=None):
    if not os.path.exists(chkp_dir):
        os.makedirs(chkp_dir)

    result_file = chkp_dir + '/' + mode + '_result.h5'

    with h5py.File(result_file, 'w') as hdf:
        for key, values in result_dict.items():
            n = len(values) if num is None else num
            for i in range(n):
                hdf.create_dataset(key + "/" + str(i), data=values[i])


def adjust_learning_rate(optimizer, epoch, base_lr, rate=0.1, period=30):
    """Sets the learning rate to the initial LR decayed by rate every period epochs"""
    lr = base_lr * (rate ** (epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
