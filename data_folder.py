# -*- coding:utf-8 -*-
import os, h5py, gc
import numpy as np
import PIL
import torch
import torch.utils.data as data
from torchvision import transforms
from scipy.misc import imresize
from transform_augment import Mask, HorizontalFlip, RandomPosCrop, Compose, ToTensor, PaddingEX2, RandomNoise


class ImageFolder(data.Dataset):

    def __init__(self, args, data_type='train'):

        dataset_path = args['data']['dataset_path']
        with h5py.File(dataset_path, 'r') as hdf:
            dataset_name = hdf.attrs['dataset_name']
            if data_type == 'train':
                raw_cnt_list = hdf['train_count'][:].tolist()
                img_name_list = hdf['train_img_name'][:]
                process_dict = args['data']['train_img_process']
            elif data_type == 'test':
                raw_cnt_list = hdf['test_count'][:].tolist()
                img_name_list = hdf['test_img_name'][:]
                process_dict = args['data']['test_img_process']
            else:
                raise Exception("Undefined data type: " + data_type)

        self.dmap_list = self._load_dataset(dataset_path, args['data']['dmap_group'], img_name_list)
        self.rmap_list = self._load_dataset(dataset_path, args['data']['rmap_group'], img_name_list)

        raw_img_path = args['data']['raw_img_path']
        img_num_channel = args['data']['img_num_channel']
        self.image_list = self._load_img(img_name_list, raw_img_path, img_num_channel)
        self.image_list, self.dmap_list, self.rmap_list = self._preprocess(process_dict)
        self.sample_number = len(self.image_list)

        print('Load dataset: {}, channel: {}, # of images: {}, # of samples: {}, label: {}'.format(\
                dataset_name, img_num_channel, len(img_name_list), self.sample_number, args['data']['dmap_group']))


    def _load_img(self, img_name_list, raw_img_path, img_num_channel):
        '''
        load PIL images, and then convert PIL images to Tensors
        return a list of 3D tensor of image
        '''
        img_list = []

        for img_name in img_name_list:
            path = raw_img_path + img_name
            img = PIL.Image.open(path)
            if img_num_channel == 1:
                img = img.convert('L')
            elif img_num_channel == 3:
                img = img.convert('RGB')

            img_list.append(img)

        return img_list


    def _load_dataset(self, dataset_path, group, name_list, mode='value'):
        with h5py.File(dataset_path, 'r') as hdf:
            data_list = [hdf[group+'/'+name][:,:] for name in name_list]
            # convert data into 3D tensor (Channel, H, W)
            data_list = [torch.from_numpy(data[np.newaxis,:,:]).float() for data in data_list]
        return data_list

    def _preprocess(self, process_dict):
        img_list, dmap_list, rmap_list = self.image_list, self.dmap_list, self.rmap_list

        to_tensor = ToTensor()
        img_list = [to_tensor(img) for img in img_list]

        if "normalize" in process_dict:
            normalizer = transforms.Normalize(mean=[127, 127, 127], std=[255, 255, 255])
            img_list = [normalizer(img) for img in img_list]

        if "GaussianNoise" in process_dict:
            num_scale = len(process_dict["GaussianNoise"])
            noise = RandomNoise('GaussianNoise', scale_list=process_dict["GaussianNoise"])
            noisy_result = [noise(img) for img in img_list]

            for i, noisy_img_list in enumerate(noisy_result):
                img_list.extend(noisy_img_list)
            dmap_list.extend([dmap for dmap in dmap_list for i in range(num_scale)])
            rmap_list.extend([rmap for rmap in rmap_list for i in range(num_scale)])

        if "HorizontalFlip" in process_dict:
            flip = HorizontalFlip()
            result = [flip(img_list[i], dmap_list[i], rmap_list[i]) for i in range(len(img_list))]
            flip_img_list, flip_dmap_list, flip_rmap_list = zip(*result)
            img_list.extend(flip_img_list)
            dmap_list.extend(flip_dmap_list)
            rmap_list.extend(flip_rmap_list)

        if "RandomPosCrop" in process_dict:
            size = process_dict["RandomPosCrop"]["size"]
            number = process_dict["RandomPosCrop"]["number"]
            crop = RandomPosCrop(size, number)

            result = [crop(img_list[i], dmap_list[i], rmap_list[i]) for i in range(len(img_list))]
            img_list, dmap_list, rmap_list = [], [], []
            for (img, dmap, rmap) in result:
                img_list.extend(img)
                dmap_list.extend(dmap)
                rmap_list.extend(rmap)

        if "PaddingEX2" in process_dict:
            padding = PaddingEX2(process_dict['PaddingEX2'])
            result = [padding(img_list[i], dmap_list[i], rmap_list[i]) for i in range(len(img_list))]
            img_list, dmap_list, rmap_list = zip(*result)

        return img_list, dmap_list, rmap_list


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, dmap)
        """
        img = self.image_list[index]
        dmap = self.dmap_list[index]
        rmap = self.rmap_list[index]

        return (index, img, dmap, rmap)

    def __len__(self):
        """
        return number of samples
        """
        return self.sample_number
