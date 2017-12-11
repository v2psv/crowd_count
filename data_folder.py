# -*- coding:utf-8 -*-
import os, h5py, gc, random
import numpy as np
import PIL, numbers
import torch
import torch.utils.data as data
from torchvision import transforms
from scipy.misc import imresize
from transform_augment import ToTensor, PaddingEX2

def load_img(path, img_num_channel):
    if img_num_channel == 1:
        return PIL.Image.open(path).convert('L')
    elif img_num_channel == 3:
        return PIL.Image.open(path).convert('RGB')

def pad2d_max(tensor, height, width):
    '''
    input:
        tensor: 3D tensor, C*H*W
        height, width: maximum height and width,
        if H larger than height, then it's no need to padding
    '''
    c, h, w = tensor.size()
    pad_h, pad_w = max(height - h, 0), max(width - w, 0)
    if pad_h == 0 and pad_w == 0:
        return tensor.float()
    else:
        p_tensor = torch.zeros(c, h+pad_h, w+pad_w)
        p_tensor[:, 0:h, 0:w] = tensor[:,:,:]
        return p_tensor.float()

def get_patches(idx, pos, patch_h, patch_w):
    '''
    get positions of patches
    return: N*5 array: image_id, left, top, right, bottom
    '''
    patches = np.zeros((pos.shape[0], 5))
    patches[:, 0] = idx
    patches[:, 1:3] = pos[:, :]
    patches[:, 3] = pos[:, 0] + patch_w
    patches[:, 4] = pos[:, 1] + patch_h

    return patches.astype(int)


class DataFolder(data.Dataset):
    def __init__(self, args, mode='train'):
        assert mode in ['train', 'test']
        assert args['model']['target'] in ['Density_Context', 'Perspective']

        self.target = args['model']['target']
        self.mode = mode

        dataset_path = args['data']['dataset_path']
        raw_img_path = args['data']['raw_img_path']
        img_num_channel = args['data']['img_num_channel']
        patch_size = args['data']['patch_size']
        downscale = args['data']['downscale']

        with h5py.File(dataset_path, 'r') as hdf:
            dataset_name = hdf.attrs['dataset_name']
            img_name_list = hdf[mode]['img_name'][:]
            if self.mode == 'train':
                position_list = [hdf['train']['patch_position'][name][...] for name in img_name_list]

        if self.target == 'Density_Context':
            if self.mode == 'train':
                self.image_list, self.image_patch = self._load_img(img_name_list, raw_img_path, img_num_channel, position_list, patch_size)
                self.density_list, self.density_patch = self._load_dataset(dataset_path, 'train/'+args['data']['density_group'], img_name_list, position_list, patch_size, downscale)
                self.perspective_list, self.perspective_patch = self._load_dataset("./dataset/pred_perspective.h5", 'train/pred_perspective', img_name_list, position_list, patch_size, downscale)
                context_list = self._load_dataset(dataset_path, 'train/'+args['data']['context_group'], img_name_list)
                self.context_map = torch.cat(context_list, 0)
                self.sample_number = self.image_patch.shape[0]
            elif self.mode == 'test':
                self.image_list = self._load_img(img_name_list, raw_img_path, img_num_channel)
                self.density_list = self._load_dataset(dataset_path, 'test/'+args['data']['density_group'], img_name_list)
                self.perspective_list = self._load_dataset("./dataset/pred_perspective.h5", 'test/pred_perspective', img_name_list)
                self.context_list = self._load_dataset(dataset_path, 'test/'+args['data']['context_group'], img_name_list)
                self.sample_number = len(self.image_list)
                padding = PaddingEX2(16)
                result = [padding(self.image_list[i], [self.density_list[i], self.context_list[i], self.perspective_list[i]], ratio=downscale) for i in range(self.sample_number)]
                self.image_list, self.density_list, self.context_list, self.perspective_list = zip(*result)
        elif self.target == 'Perspective':
                self.image_list = self._load_img(img_name_list, raw_img_path, img_num_channel)
                self.perspective_list = self._load_dataset(dataset_path, self.mode+'/'+args['data']['perspective_group'], img_name_list)
                self.sample_number = len(self.image_list)
                padding = PaddingEX2(16)
                result = [padding(self.image_list[i], [self.perspective_list[i]], ratio=downscale) for i in range(self.sample_number)]
                self.image_list, self.perspective_list = zip(*result)
        self.image_number = len(self.image_list)
        print('Load dataset: {}, channel: {}, # of images: {}, # of samples: {}'.format(\
                dataset_name, img_num_channel, self.image_number, self.sample_number))


    def _load_img(self, img_name_list, raw_img_path, img_num_channel, position_list=None, patch_size=[256, 256]):
        '''
        load PIL images, and convert PIL images to patch Tensors
        '''
        img_list = [load_img(raw_img_path + name, img_num_channel) for name in img_name_list]
        to_tensor = ToTensor()
        normalizer = transforms.Normalize(mean=[127, 127, 127], std=[255, 255, 255])
        img_list = [normalizer(to_tensor(img)) for img in img_list]

        if position_list is not None:
            patch_h, patch_w = patch_size
            img_list = [pad2d_max(img, patch_h, patch_w) for img in img_list]
            image_patch = [get_patches(i, pos, patch_h, patch_w) for i, pos in enumerate(position_list)]
            image_patch = np.concatenate(image_patch, axis=0)
            # 4-D: N*C*H*W
            return img_list, image_patch
        else:
            return img_list

    def _load_dataset(self, dataset_path, group, name_list, position_list=None, patch_size=[256, 256], downscale=4):
        with h5py.File(dataset_path, 'r') as hdf:
            data_list = [hdf[group][name][...] for name in name_list]

        # convert data into 3D tensor (C, H, W)
        if len(data_list[0].shape) == 2:
            data_list = [data[np.newaxis,:,:] for data in data_list]
        data_list = [torch.from_numpy(data).float() for data in data_list]

        if position_list is not None:
            patch_h, patch_w = patch_size[0] / downscale, patch_size[1] / downscale
            data_list = [pad2d_max(data, patch_h, patch_w) for data in data_list]
            data_patch = [get_patches(i, pos/downscale, patch_h, patch_w) for i, pos in enumerate(position_list)]
            data_patch = np.concatenate(data_patch, axis=0)
            return data_list, data_patch
        else:
            return data_list

    def _get_train_data(self, index):
        if self.target == 'Density_Context':
            pos = self.image_patch[index,:]
            image = self.image_list[pos[0]][:,pos[2]:pos[4],pos[1]:pos[3]]
            if random.random() < 0.2:
                image = image + image.new(image.size()).normal_(0, 0.03)

            pos = self.density_patch[index, :]
            density = self.density_list[pos[0]][:,pos[2]:pos[4],pos[1]:pos[3]]

            pos = self.perspective_patch[index, :]
            perspective = self.perspective_list[pos[0]][:,pos[2]:pos[4],pos[1]:pos[3]]

            context = self.context_map[index, :, :].unsqueeze(0)

            return (index, image, density, context, perspective)

        elif self.target == 'Perspective':
            image = self.image_list[index]
            if random.random() < 0.5:
                image = image + image.new(image.size()).normal_(0, 0.03)
            perspective = self.perspective_list[index]
            return (index, image, perspective)

    def _get_test_data(self, index):
        if self.target == 'Density_Context':
            image = self.image_list[index]
            density = self.density_list[index]
            context = self.context_list[index]
            perspective = self.perspective_list[index]
            # print image.size(), density.size(), context.size(), perspective.size()
            return (index, image, density, context, perspective)
        elif self.target == 'Perspective':
            image = self.image_list[index]
            perspective = self.perspective_list[index]
            return (index, image, perspective)

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
