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

<<<<<<< HEAD
    def __init__(self, args, type='train'):
        self.args = args
        self.type = type
=======
    def __init__(self, args, idx_list):
        self.args = args
        self.idx_list = sorted(idx_list)
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
        self.dataset_path = args['data']['dataset_path']
        self.raw_img_path = args['data']['raw_img_path']
        self.dmap_group = args['data']['dmap_group']
        self.img_num_channel = args['data']['img_num_channel']

<<<<<<< HEAD
        with h5py.File(self.dataset_path, 'r') as hdf:
            self.dataset_name = hdf.attrs['dataset_name']
            if self.type == 'train':
                self.idx_list = sorted(hdf['train_idx'][:])
                self.augmentor = self._get_augmentor(args['data']['img_augment'])
            else:
                self.idx_list = sorted(hdf['test_idx'][:])

=======
        with h5py.File(self.dataset_path) as hdf:
            self.dataset_name = hdf.attrs['dataset_name']
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
            self.cnt_list = hdf['raw_count'][self.idx_list].tolist()
            self.img_name_list = hdf['img_name_list'][self.idx_list]

            self.dmap_list = [hdf[self.dmap_group+'/'+img_name][:,:] for img_name in self.img_name_list]
            self.dmap_list = [torch.from_numpy(dmap[np.newaxis,:,:]).float() for dmap in self.dmap_list]

<<<<<<< HEAD
        self.image_list = self._load_img()

        if self.type == 'train':
            # 'train_img_141.jpg' has a bug
            self.image_list[140] = self.image_list[140][:, :-5, :-5]
            self.dmap_list[140] = self.dmap_list[140][:, :-23, :-20]
=======
            self.pmap_list = [hdf['pmap/'+img_name][:,:] for img_name in self.img_name_list]
            self.pmap_list = [torch.from_numpy(pmap[np.newaxis,:,:]).float() for pmap in self.pmap_list]

        self.image_list = self._load_img()
        self.augmentor = self._get_augmentor(args['data']['img_augment'])
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b

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
<<<<<<< HEAD
        cnt = self.cnt_list[index]
        img_name = self.img_name_list[index]

        if self.type == 'train':
            img, dmap = self.augmentor(img, dmap)

        return (idx, img, dmap, cnt)
=======
        pmap = self.pmap_list[index]
        cnt = self.cnt_list[index]
        img_name = self.img_name_list[index]

        img, dmap, pmap = self.augmentor(img, dmap, pmap)

        # print(img.size(), dmap.size(), pmap.size())
        return (idx, img, dmap, cnt, pmap)
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b

    def __len__(self):
        """
        return number of samples
        """
        return len(self.img_name_list)
