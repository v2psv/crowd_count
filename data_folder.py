# -*- coding:utf-8 -*-
import os, h5py, gc, random
import numpy as np
import PIL, numbers
import random, itertools
from scipy.spatial.distance import pdist
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from scipy.misc import imresize
from scipy.signal import convolve2d
from scipy.stats import itemfreq
from transform_augment import ToTensor, PaddingEX2, pad_2d

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

class DataFolder(data.Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'test']

        self.mode = mode
        self.target = args['model']['target']
        self.random_noise = args['data'][mode]['random_noise']

        if mode == 'train':
            self._load_train_data(args)
        elif mode == 'test':
            self._load_test_data(args)

        gc.collect()

    def _load_train_data(self, args):
        dataset_path = args['data']['dataset_path']
        raw_img_path = args['data']['raw_img_path']
        downscale = args['data']['downscale']
        patch_size = np.array(args['data']['train']['patch_size'])
        num_patch = args['data']['train']['num_patch']

        with h5py.File(dataset_path, 'r') as hdf:
            dataset_name = hdf.attrs['dataset_name']
            img_size = hdf['train']['img_size'][:,:]
            img_name_list = hdf['train']['img_name'][:]

        # self.loc_list = self.load_location(dataset_path, 'train/location/', img_name_list)
        self.image_list_r, self.image_list_g = self.load_img(img_name_list, raw_img_path, min_size=patch_size)
        if args['data']['use_roi']:
            self.image_list_r = self.apply_roi(self.image_list_r, dataset_path, 'train/roi', img_name_list, min_size=patch_size)
            # self.image_list_g = self.apply_roi(self.image_list_g, dataset_path, 'train/roi', img_name_list, min_size=patch_size)

        self.image_patch, self.label_patch = self.get_patches(img_size, patch_size, num_patch, downscale)
        self.sample_number = self.image_patch.shape[0]
        self.image_number = len(self.image_list_r)

        if self.target == 'Density':
            self.density_list = self.load_dataset(dataset_path, 'train/'+args['data']['density_group'], img_name_list, min_size=patch_size/4)
        elif self.target == 'ContextPyramid':
            self.density_list = self.load_dataset(dataset_path, 'train/'+args['data']['density_group'], img_name_list, min_size=patch_size/4)
            # self.context1_map = get_patch_context(self.density_list, self.label_patch, wind_size=16/downscale)#32, levels=[1, 5, 10, 20]
            # self.context2_map = get_patch_context(self.density_list, self.label_patch, wind_size=64/downscale)#128, levels=[10, 40, 80, 160]
            # self.context1_map = get_context(self.density_list, wind_size=16/downscale, bs=100, levels=[0, 0.01, 0.5, 100], patch_pos=self.label_patch)
            self.context_map = get_context(self.density_list, wind_size=64/downscale, bs=100, levels=[0, 0.5, 2, 100], patch_pos=self.label_patch)
        elif self.target == 'PerspectContextPyramid':
            self.density_list = self.load_dataset(dataset_path, 'train/'+args['data']['density_group'], img_name_list, min_size=patch_size/4)
            self.context_map = get_context(self.density_list, wind_size=64/downscale, bs=100, levels=[0, 0.5, 2, 100], patch_pos=self.label_patch)
            self.perspective_list = self.load_dataset(dataset_path, 'train/perspective', img_name_list, min_size=patch_size/4)
        elif self.target == 'Scene':
            self.context_list = self.load_dataset(dataset_path, 'train/'+args['data']['context_group'], img_name_list, min_size=patch_size/4)
            self.perspective_list = self.load_dataset(dataset_path, 'train/'+args['data']['perspect_group'], img_name_list, min_size=patch_size/4)

        print('Load dataset: {}, # of images: {}, # of samples: {}'.format(dataset_name, self.image_number, self.sample_number))

    def _load_test_data(self, args):
        dataset_path = args['data']['dataset_path']
        raw_img_path = args['data']['raw_img_path']
        downscale = args['data']['downscale']

        with h5py.File(dataset_path, 'r') as hdf:
            dataset_name = hdf.attrs['dataset_name']
            img_size = hdf['test']['img_size'][:,:]
            img_name_list = hdf['test']['img_name'][:]

        # self.loc_list = self.load_location(dataset_path, 'test/location/', img_name_list)
        self.image_list, _ = self.load_img(img_name_list, raw_img_path)
        self.sample_number = len(self.image_list)
        self.image_number = len(self.image_list)

        self.image_list_r, _ = self.load_img(img_name_list, raw_img_path)
        if args['data']['use_roi']:
            self.image_list_r = self.apply_roi(self.image_list_r, dataset_path, 'test/roi', img_name_list)
        padding = PaddingEX2(32)
        self.image_list = [padding(image) for image in self.image_list]
        label_min_size = [np.array((img.size(1)//downscale, img.size(2)//downscale)) for img in self.image_list]

        if self.target == 'Context':
            self.context_list = get_context(self.density_list, wind_size=64/downscale, bs=100, levels=[2, 5, 20, 40, 80])
            self.context_list = self.load_dataset(dataset_path, 'test/'+args['data']['context_group'], img_name_list, min_size=label_min_size)
        elif self.target == 'ContextPyramid':
            self.density_list = self.load_dataset(dataset_path, 'test/'+args['data']['density_group'], img_name_list, min_size=label_min_size)
            # self.context1_list = get_context(self.density_list, wind_size=16/downscale, bs=100, levels=[0, 0.01, 0.5, 100])
            self.context_list = get_context(self.density_list, wind_size=64/downscale, bs=100, levels=[0, 0.5, 2, 100])
        elif self.target == 'Density':
            self.density_list = self.load_dataset(dataset_path, 'test/'+args['data']['density_group'], img_name_list, min_size=label_min_size)
        elif self.target == 'Perspect':
            self.perspective_list = self.load_dataset(dataset_path, 'test/'+args['data']['perspect_group'], img_name_list, min_size=label_min_size)
            # self.perspective_list = [self.value2class(p, [0.02, 0.1, 0.2, 0.5]) for p in self.perspective_list]
        elif self.target == 'Scene':
            self.context_list = self.load_dataset(dataset_path, 'test/'+args['data']['context_group'], img_name_list, min_size=label_min_size)
            self.perspective_list = self.load_dataset(dataset_path, 'test/'+args['data']['perspect_group'], img_name_list, min_size=label_min_size)
        elif self.target == 'MultiTask':
            self.density_list = self.load_dataset(dataset_path, 'test/'+args['data']['density_group'], img_name_list, min_size=label_min_size)
            self.context_list = get_context_list(self.density_list, wind_size=64/downscale, levels=[5, 20, 40, 80])

        print('Load dataset: {}, # of images: {}, # of samples: {}'.format(dataset_name, self.image_number, self.sample_number))

    def get_patches(self, img_size, patch_size, num_patch, downscale):
        # img_size_list = [(img.size(1), img.size(2)) for img in img_list]
        # result = [get_patch(idx, img_size, patch_size, num_patch, mode='random', downscale=downscale) for idx, img_size in enumerate(img_size_list)]
        result = [get_patch(idx, img_size[idx, :], patch_size, num_patch, mode='random', downscale=downscale) for idx in range(img_size.shape[0])]
        image_patch, label_patch = zip(*result)
        image_patch = np.concatenate(image_patch, axis=0)
        label_patch = np.concatenate(label_patch, axis=0)
        return image_patch, label_patch

    def convert_label(self, data, mode='class', bins=None):
        assert mode in ['class', 'order', 'log']
        if mode == 'class':
            data[data<bins[0]] = 0
            data[data>=bins[-1]] = len(bins)
            for i in range(n-1):
                data[(data>=bins[i]) & (data<bins[i+1])] = i+1
        elif mode == 'log':
            data = torch.log(data+1)
        elif mode == 'order':
            data = data.numpy()[0,:,:]
            order = np.zeros((len(bins), data.shape[0], data.shape[1]))
            for i in range(n):
                order[i, :, :][data>=bins[i]] = 1
            data = torch.from_numpy(order)

        return data

    def load_img(self, img_name_list, raw_img_path, min_size=None, gray_scale=False):
        '''
        load PIL images, and convert PIL images to patch Tensors
        '''
        def load(path):
            return PIL.Image.open(path).convert('RGB')

        print(f"Loading images...\r",end="")

        num = len(img_name_list)
        to_tensor = ToTensor()
        normalizer = transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])

        img_list_r = [load(raw_img_path + name.decode('UTF-8')) for name in img_name_list]
        img_list_r = [normalizer(to_tensor(img)) for img in img_list_r]

        """
        if gray_scale:
            img_list_g = [img.convert('L').convert('RGB') for img in img_list_r]
        """
        if min_size is not None:
            min_h, min_w = min_size[0], min_size[1]
            img_list_r = [pad2d_max(data, min_h, min_w) for data in img_list_r]
            # img_list_g = [pad2d_max(data, min_h, min_w) for data in img_list_g]
        return img_list_r, None

    def apply_roi(self, img_list, dataset_path, group, name_list, min_size=None):
        num = len(img_list)

        with h5py.File(dataset_path, 'r') as hdf:
            for i, name in enumerate(name_list):
                print(f"Applying ROI... {i/num*100:.1f} % ({i} of {num})\r",end="")
                roi = hdf[group][name][...]
                roi = np.repeat(roi[np.newaxis,:,:], 3, axis=0)
                img_list[i] = torch.from_numpy(roi).float() * img_list[i]
            print(" "*100+"\r", end="")
        return img_list

    def load_dataset(self, dataset_path, group, name_list, min_size=None):
        num = len(name_list)

        data_list = []
        with h5py.File(dataset_path, 'r') as hdf:
            for i, name in enumerate(name_list):
                data = hdf[group][name][...]
                if len(data.shape) == 2:
                    data = data[np.newaxis,:,:]
                data_list.append(torch.from_numpy(data).float())
                if i % 20 == 0:
                    print(f"Loading {group}... {i/num*100:.1f} % ({i} of {num})\r",end="")

        if min_size is not None:
            if len(min_size) == num:
                min_h, min_w = min_size[0], min_size[1]
                data_list = [pad2d_max(data, min_size[i][0], min_size[i][1]) for i, data in enumerate(data_list)]
            else:
                data_list = [pad2d_max(data, min_size[0], min_size[1]) for data in data_list]
        print(" "*100+"\r", end="")
        return data_list

    def load_location(self, dataset_path, group, name_list):
        num = len(name_list)
        with h5py.File(dataset_path, 'r') as hdf:
            loc_list = [hdf[group][name][:,:].astype(int) for name in name_list]
        return loc_list

    def load_dist_histogram(self, dataset_path, histogram_group, name_list):
        num = len(name_list)
        data = np.zeros((num*100, 6))
        with h5py.File(dataset_path, 'r') as hdf:
            for i, name in enumerate(name_list):
                data[i*100:(i+1)*100, :, :] = hdf[histogram_group][name][...]
        data = data / 128 / 128

        def chi_square(a, b):
            return np.sum((a-b)**2/(a+b))

        similarity = pdist(data, metric=chi_square)
        sim_order = np.argsort(similarity, axis=1)

        return data, similarity

    def augment_img(self, image, **kwargs):
        if random.random() < 0.3:
            image = image + image.new(image.size()).normal_(0, 0.03)
        """
        if random.random() < 0.4:
            noise = np.random.uniform(0, 1, [image.size(0), image.size(1), image.size(2)])
            z = np.where(noise < 0.03)
            o = np.where(noise > 0.97)
            image[z] = 0.5
            image[o] = -0.5
        """
        ret = []
        if random.random() < 0.5:
            ret.append(torch.from_numpy(np.flip(image.numpy(), 2).copy()))
            for name, label in kwargs.items():
                ret.append(torch.from_numpy(np.flip(label.numpy(), 2).copy()))
        else:
            ret.append(image)
            for name, label in kwargs.items():
                ret.append(label)

        return ret

    def _get_train_data(self, index):
        i, l, u, r, b = self.image_patch[index,:]
        """
        if random.random() < 10:
            image = self.image_list_r[i][:, u:b, l:r]
        else:
            image = self.image_list_g[i][:, u:b, l:r]
        """
        image = self.image_list_r[i][:, u:b, l:r]
        out = [index]

        i, l, u, r, b = self.label_patch[index,:]
        if self.target == 'Density':
            label = self.density_list[i][:, u:b, l:r]
            image, label = self.augment_img(image, density=label)
            out.extend([image, label])

        elif self.target == 'Context':
            label = self.context_map[index, :, :].unsqueeze(0)
            image, label = self.augment_img(image, context=label)
            out.extend([image, label])

        elif self.target == 'ContextPyramid':
            density = self.density_list[i][:, u:b, l:r]
            context = self.context_map[index, :, :]
            image, density, context = self.augment_img(image, density=density, context=context)
            out.extend([image, density, context])

        elif self.target == 'Perspect':
            label = self.perspective_list[i][:, u:b, l:r]
            image, label = self.augment_img(image, perspect=label)
            out.extend([image, label])

        elif self.target == 'Scene':
            context = self.context_map[index, :, :].unsqueeze(0)
            perspect = self.perspective_list[i][:, u:b, l:r]
            image, context, perspect = self.augment_img(image, context=context, perspect=perspect)
            out.extend([image, context, perspect])

        return out

    def _get_test_data(self, index):
        image = self.image_list[index]
        out = [index, image]

        if self.target == 'Density':
            out.append(self.density_list[index])
        elif self.target == 'Context':
            out.append(self.context_list[index])
        elif self.target == 'ContextPyramid':
            out.append(self.density_list[index])
            out.append(self.context_list[index])
        elif self.target == 'Perspect':
            out.append(self.perspective_list[index])
        elif self.target == 'Scene':
            out.append(self.context_list[index])
            out.append(self.perspective_list[index])

        return out

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, dmap)
        """
        if self.mode == 'train':
            return self._get_train_data(index)
        elif self.mode == 'test':
            return self._get_test_data(index)

    def __len__(self):
        """
        return number of samples
        """
        return self.sample_number
