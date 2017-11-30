# -*- coding:utf-8 -*-
import os, h5py, gc
import numpy as np
import PIL
import torch
import torch.utils.data as data
from torchvision import transforms
from scipy.misc import imresize
from transform_augment import Mask, HorizontalFlip, RandomPosCrop, Compose, ToTensor, PaddingEX2


class ImageFolder(data.Dataset):

    def __init__(self, args, data_type='train'):
        self.args = args
        self.dataset_path = args['data']['dataset_path']
        self.raw_img_path = args['data']['raw_img_path']
        self.dmap_group = args['data']['dmap_group']
        self.img_num_channel = args['data']['img_num_channel']

        with h5py.File(self.dataset_path, 'r') as hdf:
            self.dataset_name = hdf.attrs['dataset_name']
            if data_type == 'train':
                self.raw_cnt_list = hdf['train_count'][:].tolist()
                img_name_list = hdf['train_img_name'][:]
                process_dict = self.args['data']['train_img_process']
            elif data_type == 'test':
                self.raw_cnt_list = hdf['test_count'][:].tolist()
                img_name_list = hdf['test_img_name'][:]
                process_dict = self.args['data']['test_img_process']
            else:
                raise Exception("Undefined data type: " + data_type)

            dmap_list = [hdf[self.dmap_group+'/'+img_name][:,:] for img_name in img_name_list]
            dmap_list = [torch.from_numpy(dmap[np.newaxis,:,:]).float() for dmap in dmap_list]

        raw_img_list = self._load_img(img_name_list)
        self.image_list, self.dmap_list = self._preprocess(raw_img_list, dmap_list, process_dict)
        self.sample_number = len(self.image_list)

        print('Load dataset: {}, channel: {}, # of images: {}, # of samples: {}, label: {}'.format(\
            self.dataset_name, self.img_num_channel, len(img_name_list), self.sample_number, self.dmap_group))


    def _load_img(self, img_name_list):
        '''
        load PIL images, and then convert PIL images to Tensors
        return a list of 3D tensor of image
        '''
        img_list = []

        for img_name in img_name_list:
            path = self.raw_img_path + img_name
            img = PIL.Image.open(path)
            if self.img_num_channel == 1:
                img = img.convert('L')
            elif self.img_num_channel == 3:
                img = img.convert('RGB')

            img_list.append(img)

        return img_list

    def _preprocess(self, img_list, dmap_list, process_dict):
        to_tensor = ToTensor()
        img_list = [to_tensor(img) for img in img_list]

        if "normalize" in process_dict:
            normalizer = transforms.Normalize(mean=[127, 127, 127], std=[255, 255, 255])
            img_list = [normalizer(img) for img in img_list]

        if "HorizontalFlip" in process_dict:
            flip = HorizontalFlip()
            result = [flip(img_list[i], dmap_list[i]) for i in range(len(img_list))]
            flip_img_list, flip_dmap_list = zip(*result)
            img_list.extend(flip_img_list)
            dmap_list.extend(flip_dmap_list)

        if "PaddingEX2" in process_dict:
            padding = PaddingEX2(process_dict['PaddingEX2'])
            result = [padding(img_list[i], dmap_list[i]) for i in range(len(img_list))]
            img_list, dmap_list = zip(*result)

        if "RandomPosCrop" in process_dict:
            size = process_dict["RandomPosCrop"]["size"]
            number = process_dict["RandomPosCrop"]["number"]
            crop = RandomPosCrop(size, number)

            result = [crop(img_list[i], dmap_list[i]) for i in range(len(img_list))]
            img_list, dmap_list = [], []
            for (img, dmap) in result:
                img_list.extend(img)
                dmap_list.extend(dmap)

        return img_list, dmap_list


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, dmap)
        """
        img = self.image_list[index]
        dmap = self.dmap_list[index]

        return (index, img, dmap)

    def __len__(self):
        """
        return number of samples
        """
        return self.sample_number
