# -*- coding:utf-8 -*-
import h5py, os, sys
import json
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
from shutil import copyfile
import torch.nn.functional as F
from scipy.stats import itemfreq


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
    if 'grad_loss' in recorder:
        str = str + 'Gradient Loss {grad_loss.avg:.5f}   '.format(grad_loss=recorder['grad_loss'])
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


def pad2d_max(tensor, height, width):
    '''
    input:
        tensor: 3D tensor, C*H*W
        height, width: maximum height and width,
        if H larger than height, then it's no need to padding
    '''
    assert tensor.dim() == 2 or tensor.dim() == 3

    if tensor.dim() == 3:
        c, h, w = tensor.size()
    elif tensor.dim() == 2:
        h, w = tensor.size()

    pad_h, pad_w = int(max(height - h, 0)), int(max(width - w, 0))

    if pad_h == 0 and pad_w == 0:
        return tensor.float()

    if tensor.dim() == 3:
        p_tensor = torch.zeros(c, h+pad_h, w+pad_w)
        p_tensor[:, 0:h, 0:w] = tensor[:,:,:]
        return p_tensor.float()
    elif tensor.dim() == 2:
        p_tensor = torch.zeros(h+pad_h, w+pad_w)
        p_tensor[0:h, 0:w] = tensor[:,:]
        return p_tensor.float()

def get_patch(idx, img_size, patch_size, num_patch, mode='random', downscale=4):
    assert mode in ['random', 'evenly']
    # return: N*5 array: image_id, left, top, right, bottom
    def linspace(max_value, wind_size):
        if max_value <= wind_size:
            return [0]
        if max_value <= wind_size + num_patch:
            return [0, max_value - wind_size]
        return np.linspace(0, max_value - wind_size, num_patch).astype(int)

    img_h, img_w = img_size
    patch_h, patch_w = patch_size

    if mode == 'random':
        if img_w <= patch_w or img_h <= patch_h:
            num_patch = 1
        x = np.random.randint(0, max(1, img_w-patch_w), num_patch)
        y = np.random.randint(0, max(1, img_h-patch_h), num_patch)
        pos = np.array(list(zip(x, y)))
    elif mode == 'evenly':
        x = linspace(img_w, num_patch)
        y = linspace(img_h, num_patch)
        pos = np.array(list(itertools.product(x, y)))

    image_patch = np.zeros((pos.shape[0], 5))
    image_patch[:, 0] = idx
    image_patch[:, 1:3] = pos[:, :]
    image_patch[:, 3] = pos[:, 0] + patch_w
    image_patch[:, 4] = pos[:, 1] + patch_h

    label_patch = image_patch.copy()
    label_patch[:, 1:] = label_patch[:, 1:] / downscale


    return image_patch.astype(int), label_patch.astype(int)

def get_context(density_list, wind_size, bs=100, levels=None, patch_pos=None):
    num_img = len(density_list)
    wind_h = wind_w = int(wind_size)

    def context_convolve(data):
        kernel = torch.autograd.Variable(torch.ones(1, 1, wind_h, wind_w)).type(torch.FloatTensor)
        data = torch.autograd.Variable(torch.from_numpy(data)).type(torch.FloatTensor)
        data = F.conv2d(data, kernel, bias=None, stride=1, padding=(int(wind_h/2), int(wind_w/2)), dilation=1, groups=1)
        # data = F.avg_pool2d(data, (wind_h, wind_w), stride=1, padding=(int(wind_h//2), int(wind_w//2)), ceil_mode=True)
        data = data.data.numpy()[:,:,:-1,:-1]
        return data

    if patch_pos is None:
        img_size = np.array([(density.size(1), density.size(2)) for density in density_list]).astype(int)
        max_h, max_w = np.max(img_size, axis=0)
        context = np.zeros((num_img, 1, max_h, max_w))
        for i, loc in enumerate(density_list):
            height, width = img_size[i,:]
            context[i, 0, :height, :width] = density_list[i][0,:,:]
    else:
        _, l, t, r, b = patch_pos[0, :]
        patch_h, patch_w = int(b-t), int(r-l)
        context = np.zeros((patch_pos.shape[0], 1, patch_h, patch_w))
        for i in range(patch_pos.shape[0]):
            idx, l, t, r, b = patch_pos[i, :]
            context[i, 0, :, :] = density_list[idx][0, t:b, l:r]

    n = int(np.ceil(context.shape[0]/bs))
    for i in range(n):
        print(f"Calculating Context... {i/n*100:.1f} % ({i} of {n})\r",end="")
        start_idx, end_idx = i*bs, (i+1)*bs
        context[start_idx:end_idx, :, :, :] = context_convolve(context[start_idx:end_idx, :, :, :])
    print(" "*100+"\r", end="")

    if levels is not None:
        context = np.digitize(context, levels).astype(int) - 1
        print(f"window size: {wind_h}*{wind_w}\n", itemfreq(context))
    else:
        context = context / wind_h / wind_w

    if patch_pos is None:
        return [torch.from_numpy(context[i, :, :img_size[i][0], :img_size[i][1]]) for i in range(num_img)]
    else:
        return torch.from_numpy(context[:,:,:,:])
