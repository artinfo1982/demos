# -*- coding: utf-8 -*-

'''
将RGB图片(例如JPG)转变为各种YUV
'''

from scipy import ndimage
import numpy as np


def rgb2yuv(rgb):
    m = np.array([
        [0.29900, -0.16874, 0.50000],
        [0.58700, -0.33126, -0.41869],
        [0.11400, 0.50000, -0.08131]])
    yuv = np.ceil(np.dot(rgb, m))
    yuv[:, :, 1:] += 128
    yuv[yuv > 255] = 255
    return yuv
