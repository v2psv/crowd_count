# -*- coding:utf-8 -*-
import os, h5py, gc
import numpy as np
import PIL
import torch
import torch.utils.data as data
from torchvision import transforms
from scipy.misc import imresize
from transform_augment import Mask, RandomHorizontalFlip, RandomPosCrop, Compose, ToTensor


class ImageFolder(data.Dataset):

    def __init__(self, args, idx_list):
        self.args = args
        self.idx_list = sorted(idx_list)
        self.dataset_path = args['data']['dataset_path']
        self.raw_img_path = args['data']['raw_img_path']
        self.dmap_group = args['data']['dmap_group']
        self.img_num_channel = args['data']['img_num_channel']

        with h5py.File(self.dataset_path) as hdf:
            self.dataset_name = hdf.attrs['dataset_name']
            self.cnt_list = hdf['raw_count'][self.idx_list].tolist()
            self.img_name_list = hdf['img_name_list'][self.idx_list]

            self.dmap_list = [hdf[self.dmap_group+'/'+img_name][:,:] for img_name in self.img_name_list]
            self.dmap_list = [torch.from_numpy(dmap[np.newaxis,:,:]).float() for dmap in self.dmap_list]

            self.pmap_list = [hdf['pmap/'+img_name][:,:] for img_name in self.img_name_list]
            self.pmap_list = [torch.from_numpy(pmap[np.newaxis,:,:]).float() for pmap in self.pmap_list]

        self.image_list = self._load_img()
        self.augmentor = self._get_augmentor(args['data']['img_augment'])

        print('Load dataset: {}, channel: {}, # of images: {}, dmap group: {}'.format(\
            self.dataset_name, self.img_num_channel, len(self.img_name_list), self.dmap_group))


    def _load_img(self):
        '''
        load PIL images, and then convert PIL images to Tensors
        return a list of 3D tensor of image
        '''
        img_list = []

        processer = self._get_processor(self.args['data']['img_preprocess'])

        for img_name in self.img_name_list:
            path = self.raw_img_path + img_name
            img = PIL.Image.open(path)
            if self.img_num_channel == 1:
                img = img.convert('L')
            elif self.img_num_channel == 3:
                img = img.convert('RGB')

            img_list.append(processer(img))

        return img_list


    def _get_processor(self, process_dict):
        proc = []

        if process_dict['scale']:
            h, w = process_dict['scale']['height'], process_dict['scale']['width']
            proc.append(transforms.Scale((w, h), interpolation=PIL.Image.BILINEAR))

        proc.append(ToTensor())

        if process_dict['normalize']:
            proc.append(transforms.Normalize(mean=[127, 127, 127],
                                             std=[255, 255, 255]))

        return Compose(proc)


    def _get_augmentor(self, augment_dict):
        aug = []

        if augment_dict["RandomHorizontalFlip"]:
            aug.append(RandomHorizontalFlip())

        if "RandomPosCrop" in augment_dict:
            size = augment_dict["RandomPosCrop"]["size"]
            aug.append(RandomPosCrop(size))

        return Compose(aug)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, dmap)
        """
        idx = self.idx_list[index]
        img = self.image_list[index]
        dmap = self.dmap_list[index]
        pmap = self.pmap_list[index]
        cnt = self.cnt_list[index]
        img_name = self.img_name_list[index]

        img, dmap, pmap = self.augmentor(img, dmap, pmap)

        # print(img.size(), dmap.size(), pmap.size())
        return (idx, img, dmap, cnt, pmap)

    def __len__(self):
        """
        return number of samples
        """
        return len(self.img_name_list)