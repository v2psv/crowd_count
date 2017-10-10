import random
import numbers
import torch
import numpy as np
from scipy.misc import imresize


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, dmap=None):
        if dmap is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, dmap = t(img, dmap)
            return img, dmap


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

    def __call__(self, img, dmap):
        """
        Args:
            img (Tensor, CHW): Image to be flipped.
        Returns:
            Tensor: Randomly flipped image.
        """
        if random.random() < 0.5:
            img_np = np.flip(img.numpy(), 2).copy()
            dmap_np = np.flip(dmap.numpy(), 2).copy()
            return torch.from_numpy(img_np), torch.from_numpy(dmap_np)
        return img, dmap


class RandomPosCrop(object):
    """Crop the given image and density map at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image.
            Default is False, i.e no padding.
            If padding = 0, padding to original size.
            If a sequence of length 4 is provided, it is used to pad [left, top, right, bottom] borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding

    def __call__(self, img, dmap):
        """
        Args:
            img (Tensor): image to be cropped.
            dmap (Tensor): density map to be cropped.
        Returns:
            (Tensor, Tensor): Cropped image and dmap.
        """
        if random.random() < 0.5:
            return img, dmap

        c1, h1, w1 = img.size()
        c2, h2, w2 = dmap.size()

        # get size of cropped dmap
        ratio = int(w1 / w2)
        th1, tw1 = self.size
        th2, tw2 = int(th1 / ratio), int(tw1 / ratio)

        if w1 == tw1 and h1 == th1:
            return img, dmap

        x2 = random.randint(0, w2 - tw2)
        y2 = random.randint(0, h2 - th2)
        x1, y1 = int(x2*ratio), int(y2*ratio)

        if self.padding == 0:
            img1  = img[:, y1:y1+th1, x1:x1+tw1]
            dmap1 = dmap[:, y2:y2+th2, x2:x2+tw2]
        elif self.padding == 1:
            img1, dmap1 = torch.zeros(img.size()), torch.zeros(dmap.size())
            img1[:, :th1, :tw1]  = img[:, y1:y1+th1, x1:x1+tw1]
            dmap1[:, :th2, :tw2] = dmap[:, y2:y2+th2, x2:x2+tw2]
        elif len(self.padding) == 4:
            img1 = img
            dmap1 = dmap
        else:
            raise Exception("cannot recognize the padding method: " + str(self.padding))

        print(img1.size(), dmap1.size())
        return img1, dmap1
