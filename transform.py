import random
import numbers
import numpy as np
from scipy.misc import imresize


class Transform(object):
    """
    image transforming
    Args:
        img: image (numpy.ndarray)
        roi: roi mask, 0 for background, 1 for foreground

    Return:
        transformed image
    """
    def __init__(self, transform_list, roi=None):
        self.transform_list = transform_list
        self.roi = roi

    def __call__(self, img):
        if self.transform_list is None:
            return img

        for trans in self.transform_list:
            if 'mask' in trans:
                _, fill = trans.split('_')
                img = self._mask_roi(img, self.roi, fill=int(fill))

            elif 'normalize' in trans:
                _, mean, std = trans.split('_')
                img = self._normalize(img, int(mean), int(std))

            elif 'scale' in trans:
                size = trans.split('_')[1:]
                size = [int(x) for x in size]
                img = self._scale(img, size)

            else:
                print('Undefined transform: ' + trans)

        return img

    def _mask_roi(self, img, roi, fill=0):
        if self.roi is None:
            return img

        h1, w1 = img.shape
        h2, w2 = roi.shape
        if h1 != h2 or w1 != w2:
            print('The ROI will be resized to match the size of image...')
            roi = imresize(roi, (h1, w1))

        if fill == 0:
            img = img * roi
        else:
            img = img * roi + (roi == 0) * fill
        return img

    def _normalize(self, img, mean, std):
        if (mean == 0) and (std == 0):
            mean, std = img.mean(), img.std()
        else:
            mean, std = float(mean), float(std)
        return (img - mean) / std

    def _scale(self, img, size):
        """
        size: number or list/tuple (height, width)
        """
        if isinstance(size, numbers.Number):
            size = (size, size)
        return imresize(img, size)


class DataAugment(object):
    """
    data augmentation
    Args:
    Returns:
        img, dmap: processed image and dmap
    """
    def __init__(self, augment_list=None, pad=True):
        self.augment_list = augment_list
        self.pad = pad

    def __call__(self, img, dmap):
        if self.augment_list is None:
            return img, dmap

        t = np.random.randint(1, 3)
        for i in range(t):
            method = random.choice(self.augment_list)
            img, dmap = self._generate_img_dmap(method, img, dmap)

        return img, dmap


    def _generate_img_dmap(self, method, img, dmap):
        if method == 'None':
            return img, dmap

        elif 'Add' in method:
            _, value = method.split('_')
            return self._add(img, dmap, float(value))

        elif method == 'Invert':
            return self._invert(img, dmap)

        elif method == 'HorizonFlip':
            return self._horizon_flip(img, dmap)

        elif 'RandomPosCrop' in method:
            size = method.split('_')[1:]
            size = [int(x) for x in size]
            return self._random_pos_crop(img, dmap, size)

    def _invert(self, img, dmap):
        return 1-img, dmap

    def _add(self, img, dmap, value):
        return img+value, dmap

    def _horizon_flip(self, img, dmap):
        return np.fliplr(img).copy(), np.fliplr(dmap).copy()

    def _random_pos_crop(self, img, dmap, size):
        """
        size: number or list/tuple (height, width)
        """
        if isinstance(size, numbers.Number):
            size = (size, size)
        ratio = img.shape[0] / dmap.shape[0]
        size = (int(size[0] / ratio), int(size[1] / ratio))

        h, w = dmap.shape
        x = random.randint(0, w - size[1])
        y = random.randint(0, h - size[0])

        if self.pad:
            img1, dmap1 = np.zeros(img.shape), np.zeros(dmap.shape)
            img1[:ratio*size[0], :ratio*size[1]] = img[ratio*y:ratio*(y+size[0]), ratio*x:ratio*(x+size[1])]
            dmap1[:size[0], :size[1]] = dmap[y:y+size[0], x:x+size[1]]
        else:
            img1 = img[ratio*y:ratio*(y+size[0]), ratio*x:ratio*(x+size[1])]
            dmap1 = dmap[y:y+size[0], x:x+size[1]]

        return img1, dmap1
