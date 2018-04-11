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
from utility import pad2d_max, get_patch, get_context

class DataFolder(data.Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'test']

        self.mode = mode
        self.use_pmap = args['data']['use_pmap']

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
        density_group = args['data']['density_group']

        with h5py.File(dataset_path, 'r') as hdf:
            dataset_name = hdf.attrs['dataset_name']
            parts = hdf['train']['parts'][:]
            self.part_index = hdf['train']['img_part_index'][...].tolist()
            self.img_name_list = hdf['train']['img_name'][:]
            self.img_size = hdf['train']['img_size'][...]

        self.roi_list     = self.load_dataset(dataset_path, 'train/roi', parts, min_size=patch_size)
        self.pmap_list    = self.load_dataset(dataset_path, 'train/perspective', parts, min_size=patch_size)
        self.image_list   = self.load_img(self.img_name_list, raw_img_path, min_size=patch_size)
        # self.image_list   = self.mask_image(self.image_list, self.roi_list)
        self.density_list = self.load_dataset(dataset_path, 'train/'+density_group, self.img_name_list, min_size=patch_size/4)

        self.image_patch, self.label_patch = self.get_patches(self.img_size, patch_size, num_patch, downscale)
        self.context_map = get_context(self.density_list, wind_size=64/downscale, bs=100, levels=[0, 1, 4, 100], patch_pos=self.label_patch)
        # self.density_map = get_context(self.density_list, wind_size=8/downscale, bs=100, patch_pos=self.label_patch)

        self.image_number = len(self.image_list)
        self.sample_number = self.image_patch.shape[0]

        print('Load dataset: {}, # of images: {}, # of samples: {}'.format(dataset_name, self.image_number, self.sample_number))

    def _load_test_data(self, args):
        dataset_path = args['data']['dataset_path']
        raw_img_path = args['data']['raw_img_path']
        downscale = args['data']['downscale']
        density_group = args['data']['density_group']
        patch_size = np.array(args['data']['test']['patch_size'])

        with h5py.File(dataset_path, 'r') as hdf:
            dataset_name = hdf.attrs['dataset_name']
            parts = hdf['test']['parts'][:]
            self.part_index = hdf['test']['img_part_index'][...].tolist()
            self.img_name_list = hdf['test']['img_name'][:]
            self.img_size = hdf['test']['img_size'][...]

        self.roi_list     = self.load_dataset(dataset_path, 'test/roi', parts, min_size=patch_size)
        self.pmap_list    = self.load_dataset(dataset_path, 'test/perspective', parts, min_size=patch_size)
        self.image_list   = self.load_img(self.img_name_list, raw_img_path, min_size=patch_size)
        # self.image_list   = self.mask_image(self.image_list, self.roi_list)
        self.density_list = self.load_dataset(dataset_path, 'test/'+density_group, self.img_name_list, min_size=patch_size/4)

        self.context_list = get_context(self.density_list, wind_size=64/downscale, bs=100, levels=[0, 1, 4, 100])
        self.density_list = get_context(self.density_list, wind_size=8/downscale, bs=100)

        self.sample_number = len(self.image_list)
        self.image_number = len(self.image_list)

        print('Load dataset: {}, # of images: {}, # of samples: {}'.format(dataset_name, self.image_number, self.sample_number))

    def mask_image(self, img_list, roi_list):
        num = len(img_list)

        roi_list = [roi.clone().repeat(3, 1, 1).float() for roi in roi_list]
        img_list = [img_list[i]*roi_list[self.part_index[i]] for i in range(num)]

        return img_list

    def get_patches(self, img_size, patch_size, num_patch, downscale):
        result = [get_patch(idx, img_size[idx,:], patch_size, num_patch, mode='random', downscale=downscale) for idx in range(img_size.shape[0])]
        image_patch, label_patch = zip(*result)
        image_patch = np.concatenate(image_patch, axis=0)
        label_patch = np.concatenate(label_patch, axis=0)
        return image_patch, label_patch

    def pad_data(self, data_list, min_size):
        if len(min_size) == len(data_list):
            data_list = [pad2d_max(data, min_size[i, 0], min_size[i, 1]) for i, data in enumerate(data_list)]
        elif len(min_size) == 2:
            data_list = [pad2d_max(data, min_size[0], min_size[1]) for data in data_list]
        else:
            raise Exception("Wrong shape of min_size: ", len(min_size))

        return data_list

    def load_img(self, img_name_list, raw_img_path, min_size=None):
        '''
        load PIL images, then convert to Tensors
        '''
        print(f"Loading images...\r",end="")

        def load(path):
            return PIL.Image.open(path).convert('RGB')

        to_tensor = ToTensor()
        normalizer = transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255])

        img_list = [load(raw_img_path + name.decode('UTF-8')) for name in img_name_list]
        img_list = [normalizer(to_tensor(img)) for img in img_list]

        if min_size is not None:
            img_list = self.pad_data(img_list, min_size)
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
            data_list = self.pad_data(data_list, min_size)

        print(" "*100+"\r", end="")
        return data_list

    def augment_data(self, image, **kwargs):
        if random.random() < 0.3:
            image = image + image.new(image.size()).normal_(0, 0.03)

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

    def __getitem__(self, index):
        if self.mode == 'train':
            i, l, u, r, b = self.image_patch[index,:]
            image = self.image_list[i][:, u:b, l:r]

            i, l, u, r, b = self.label_patch[index,:]
            # density = self.density_list[i][:, u:b, l:r]
            density = self.density_map[index, :, :]
            context = self.context_map[index, :, :]

            if self.use_pmap:
                pmap = self.pmap_list[self.part_index[index]]
                image, density, context, pmap = self.augment_data(image, density=density, context=context, pmap=pmap)
                out = [index, image, density, context, pmap]
            else:
                image, density, context = self.augment_data(image, density=density, context=context)
                out = [index, image, density, context]

        elif self.mode == 'test':
            out = [index]
            out.append(self.image_list[index])
            out.append(self.density_list[index])
            out.append(self.context_list[index])
            out.append(self.roi_list[self.part_index[index]][:,::4,::4])

            if self.use_pmap:
                out.append(self.pmap_list[self.part_index[index]])

        return out

    def __len__(self):
        """
        return number of samples
        """
        return self.sample_number
