# -*- coding:utf-8 -*-
import os, h5py, gc
import numpy as np
import PIL
import torch
import torch.utils.data as data
from transform import Transform, DataAugment


class ImageFolder(data.Dataset):

    def __init__(self, dataset_path, idx_list, dmap_group='dmap', transform=None, augment=None):
        self.idx_list   = sorted(idx_list)
        self._load_dataset(dataset_path, dmap_group)
        self.transformer = Transform(transform)
        self.augmentor = DataAugment(augment)
        self.image_list = [self.transformer(img) for img in self.image_list]

    def _load_dataset(self, h5_path, dmap_group):
        with h5py.File(h5_path) as hdf:
            # dataset attributes
            self.dataset_name = hdf.attrs['dataset_name']
            self.img_channel  = hdf.attrs['img_channel']
            self.img_mean = hdf.attrs['img_mean']
            self.img_std  = hdf.attrs['img_std']

            self.img_name_list = hdf['img_name_list'][self.idx_list]
            self.cnt_list = hdf['count'][self.idx_list].tolist()
            self.dmap_list  = [hdf[dmap_group+'/'+i][:,:] for i in self.img_name_list]

            if self.img_channel == 'gray':
                self.image_list = [hdf['img/'+i][:,:] for i in self.img_name_list]
            elif self.img_channel == 'rgb':
                self.image_list = [hdf['img/'+i][:,:,:] for i in self.img_name_list]
            else:
                raise Exception('Unexcepted image channel: ' + self.img_channel)

            if 'roi_resized' in hdf:
                self.roi = hdf['roi_resized'][:,:]
            else:
                self.roi = hdf['roi'][:,:]

            print('Load dataset {}, channel: {}, {} images'.format(\
                self.dataset_name, self.img_channel, len(self.img_name_list)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, dmap)
        """
        img = self.image_list[index]
        dmap = self.dmap_list[index]
        cnt = self.cnt_list[index]
        img_name = self.img_name_list[index]

        img, dmap = self.augmentor(img, dmap)
        if self.img_channel == 'gray':
            img, dmap = img[:, :, np.newaxis], dmap[:, :, np.newaxis]
        img, dmap = img.transpose((2, 0, 1)), dmap.transpose((2, 0, 1))
        img, dmap = torch.from_numpy(img).float(), torch.from_numpy(dmap).float()
        return (index, img, dmap, cnt)

    def __len__(self):
        """
        return number of samples
        """
        return len(self.img_name_list)
