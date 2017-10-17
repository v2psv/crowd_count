import random
import numbers
import torch
import numpy as np
from scipy.misc import imresize
from torch.nn.modules import padding

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, dmap=None, pmap=None):
        if dmap is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, dmap, pmap = t(img, dmap, pmap)
            return img, dmap, pmap


class ToTensor(object):
    """Convert a ``PIL.Image`` to tensor.
    Converts a PIL.Image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # PIL image mode: L, RGB
        if pic.mode == 'RGB':
            nchannel = 3
            img = np.asarray(pic)
        elif pic.mode == 'L':
            img = np.asarray(pic)[:,:,np.newaxis]
        else:
            raise Exception('Undefined mode: {}'.format(pic.mode))

        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = torch.from_numpy(img).transpose(0, 1).transpose(0, 2).contiguous()
        return img.float()

class Mask(object):
    """Mask regions within ROI with given value.
    Given roi: numpy.ndarray (height, width, 1) and fill: a number,
    """

    def __init__(self, roi):
        self.roi = roi

    def __call__(self, image):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: masked image.
        """
        if image.shape != self.roi.shape:
            print("Warning: Inconsistant shape of image and ROI: %s and %s" %
             (str(image.shape), str(self.roi.shape)))

        return image * self.roi


class RandomHorizontalFlip(object):
    """Horizontally flip the given Tensor randomly with a probability of 0.5."""

    def __call__(self, img, dmap=None, pmap=None):
        """
        Args:
            img (Tensor, CHW): Image to be flipped.
        Returns:
            Tensor: Randomly flipped image.
        """
        res = []

        if random.random() < 0.5:
            img_np = np.flip(img.numpy(), 2).copy()
<<<<<<< HEAD
            img = torch.from_numpy(img_np)

            dmap_np = np.flip(dmap.numpy(), 2).copy()
            dmap = torch.from_numpy(dmap_np)

        return img, dmap
=======
            res.append(torch.from_numpy(img_np))

            if dmap is not None:
                dmap_np = np.flip(dmap.numpy(), 2).copy()
                res.append(torch.from_numpy(dmap_np))

            if pmap is not None:
                pmap_np = np.flip(pmap.numpy(), 2).copy()
                res.append(torch.from_numpy(pmap_np))
        else:
            res = [img, dmap, pmap]

        return res
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b


class RandomPosCrop(object):
    """Crop the given image and density map at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size


    def pad_2d(self, in_matrix, size):
        # size: tuple(left, right, top, bottom)
        if isinstance(size, numbers.Number):
            l, r, t, b = (int(size), int(size), int(size), int(size))
        else:
            l, r, t, b = size

        # pad = padding.ConstantPad3d((l, r, t, b, 0, 0), 0)
        # matrix = pad(matrix)

        c, h, w = in_matrix.size()
        matrix = torch.zeros(c, h+t+b, w+l+r)
        matrix[:, t:h+t, l:w+l] = in_matrix[:,:,:]
<<<<<<< HEAD

        return matrix

=======

        return matrix
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b


    def __call__(self, img, dmap, pmap):
        """
        Args:
            img (Tensor): image to be cropped.
            dmap (Tensor): density map to be cropped.
            pmap (Tensor): perspective map to be cropped.
        Returns:
<<<<<<< HEAD
            (Tensor, Tensor): Cropped image and dmap
=======
            (Tensor, Tensor, Tensor): Cropped image and dmap, pmap.
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
        """
        # if random.random() < 0.5:
            # return img, dmap, pmap

        c1, h1, w1 = img.size()
        c2, h2, w2 = dmap.size()

        # get size of cropped dmap
        ratio = int(w1 / w2)
        th1, tw1 = self.crop_size
        th2, tw2 = int(th1 / ratio), int(tw1 / ratio)

        if w1 < tw1:
            img  = self.pad_2d(img, (0, tw1-w1, 0, 0))
<<<<<<< HEAD
            dmap = self.pad_2d(dmap, (0, tw2-w2, 0, 0))
            w1, w2 = tw1, tw2

        if h1 <= th1:
            img  = self.pad_2d(img, (0, 0, 0, th1-h1))
=======
            pmap = self.pad_2d(pmap, (0, tw1-w1, 0, 0))
            dmap = self.pad_2d(dmap, (0, tw2-w2, 0, 0))
            w1, w2 = tw1, tw2
        if h1 <= th1:
            img  = self.pad_2d(img, (0, 0, 0, th1-h1))
            pmap = self.pad_2d(pmap, (0, 0, 0, th1-h1))
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
            dmap = self.pad_2d(dmap, (0, 0, 0, th2-h2))
            h1, h2 = th1, th2

        x2 = random.randint(0, w2 - tw2)
        y2 = random.randint(0, h2 - th2)
        x1, y1 = int(x2*ratio), int(y2*ratio)

<<<<<<< HEAD
        img  = img[:, y1:y1+th1, x1:x1+tw1]
        dmap = dmap[:, y2:y2+th2, x2:x2+tw2]

        return img, dmap
=======
        img1  = img[:, y1:y1+th1, x1:x1+tw1]
        dmap1 = dmap[:, y2:y2+th2, x2:x2+tw2]
        pmap1 = pmap[:, y1:y1+th1, x1:x1+tw1]

        # print(img1.size(), dmap1.size(), pmap1.size())
        return img1, dmap1, pmap1
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
