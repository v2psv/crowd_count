import os
import numpy as np
from PIL import Image
from scipy.misc import imresize
import math, h5py


def gaussian_kernel(ksize, sigma):
    r = (ksize - 1.) / 2.
    x_range, y_range = np.ogrid[-r:r+1, -r:r+1]

    h = np.exp( -(x_range*x_range + y_range*y_range) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumofh = h.sum()
    if sumofh != 0:
        h /= sumofh
    return h


def get_dmap(location, roi, gauss_range, sigma, downscale, pmap=None):
    # location: n*2, x in [0, width], y in [0, height]
    # roi: height*width

    # filter locations only in ROI
    height, width = roi.shape
    loc = np.floor(location).astype(int)
    loc = np.clip(loc, a_min=[0,0], a_max=[width-1, height-1])
    in_roi = roi[loc[:,1], loc[:,0]]
    loc = loc[in_roi.astype(bool)]
    cnt = loc.shape[0]

    # downscale
    height, width = int(np.ceil(roi.shape[0]/downscale)), int(np.ceil(roi.shape[1]/downscale))
    roi = imresize(roi, (height, width))
    loc = np.ceil(loc/downscale).astype(int)
    loc = np.clip(loc, a_min=[0,0], a_max=[width-1, height-1])

    kernel = gaussian_kernel(gauss_range, sigma)

    pad = int(gauss_range/2)
    dmap = np.zeros((height+2*pad, width+2*pad))

    for i in range(cnt):
        xloc, yloc = loc[i, :] + pad
        if pmap is not None:
            kernel = gaussian_kernel(gauss_range, sigma/pmap[yloc, xloc])
        dmap[yloc-pad:yloc+pad+1, xloc-pad:xloc+pad+1] += kernel

    # density in padding area is ignored, maybe need to flip and add these values to dmap
    dmap = dmap[pad:-pad, pad:-pad]
    dmap = dmap * roi
    return dmap, cnt


if __name__ == '__main__':
    num_img = 2000
    downscale = 4.0
    gauss_range = 25
    sigma = 1.25
    height, width = 160, 240

    dmap = np.zeros((num_img, int(height/downscale), int(width/downscale)))
    count = np.zeros((num_img, 3))

    with h5py.File('annotation.h5', 'r') as hdf:
        roi = imresize(hdf['roi'], (height, width))
        count[:, 0] = hdf['count']

        for i in range(num_img):
            loc = hdf['location/'+str(i)]
            m, cnt = get_dmap(loc, roi, gauss_range=gauss_range, sigma=sigma, downscale=downscale)
            dmap[i, :, :] = m
            count[i, 1] = cnt
            count[i, 2] = np.sum(m)
            print('image %d, sum(density):%.2f, cnt_roi:%d, cnt:%d' % (i, count[i, 2], count[i, 1], count[i, 0]))

    with h5py.File('density_map.h5', 'w') as hdf:
        hdf.create_dataset("dmap",  data=dmap)
        hdf.create_dataset('count', data=count)



