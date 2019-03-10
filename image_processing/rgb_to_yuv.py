# -*- coding: utf-8 -*-

'''
将RGB图片(例如JPG)转变为各种YUV

YUV采样：
    4:0:0，只有Y分量，没有UV分量
    4:4:4，每1个Y对应一组UV分量
    4:2:2，每2个Y共用一组UV分量
    4:2:0，每4个Y共用一组UV分量
    4:1:1，
    
YUV存储：
    planar：简称p，存完所有Y，然后所有U，然后所有V。例如yuv420 planar = yuv420p
    semi-planar：简称sp，存完所有Y，然后UV交替存储。例如yuv420 semi-planar = yuv420sp
    packed：YUV三者交替存储，Y0U0V0Y1U1V1...。yuv420 = yuv420 packed

YUV常见格式：
yuv400
yuv444p
yuv444
yuv422p
yuv422sp
yuv422_YUYV
yuv422_UYVY
yuv422_YVYU
yuv422_VYUY
yuv420p_I420
yuv420p_YV12
yuv420sp_NV12
yuv420sp_NV21
'''

from scipy import ndimage
import numpy as np


def rgb2yuv(file_name):
    rgb = ndimage.imread(file_name)
    m = np.array([
        [0.29900, -0.16874, 0.50000],
        [0.58700, -0.33126, -0.41869],
        [0.11400, 0.50000, -0.08131]])
    yuv = np.round(np.dot(rgb, m))
    yuv[:, :, 1:] += 128
    yuv[yuv > 255] = 255
    return yuv


def rgb_to_yuv400(rgb_file, yuv_file):
    '''
    只有Y分量
    内存排布：Y0 Y1 Y2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8)
    with open(yuv_file, 'wb') as f:
        f.write(Y)


def rgb_to_yuv444p(rgb_file, yuv_file):
    '''
    Y、U、V分量1:1:1
    内存排布：Y0 Y1 Y2 ...U0 U1 U2 ...V0 V1 V2 ...
    '''
    yuv = np.array(rgb2yuv(rgb_file), dtype=np.uint8)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(U)
        f.write(V)


def rgb_to_yuv444(rgb_file, yuv_file):
    '''
    yuv444，也称为yuv444_packed
    Y、U、V分量1:1:1
    内存排布：Y0 U0 V0 ...Y1 U1 V1 ...Y2 U2 V2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    yuv_ = np.c_[Y, U, V].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422p(rgb_file, yuv_file):
    '''
    Y、U、V分量2:1:1，U、V在原来的基础上隔位采样
    内存排布：Y0 Y1 Y2 ...U0 U1 U2 ...V0 V1 V2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::2], dtype=np.uint8).flatten()
    V = np.array(V[0::2], dtype=np.uint8).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(U)
        f.write(V)


def rgb_to_yuv422sp(rgb_file, yuv_file):
    '''
    Y、U、V分量2:1:1，U、V在原来的基础上隔位采样
    先存放所有的Y，然后U、V交错存储
    内存排布：Y0 Y1 Y2 ...U0 V0 U1 V1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::2], dtype=np.uint8).flatten()
    V = np.array(V[0::2], dtype=np.uint8).flatten()
    UV = np.c_[U, V].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(UV)


def rgb_to_yuv422_YUYV(rgb_file, yuv_file):
    '''
    Y、U、V分量2:1:1，U、V在原来的基础上隔位采样
    YUV按照YUYV的组合排序
    内存排布：Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::2], dtype=np.uint8).flatten()
    V = np.array(V[0::2], dtype=np.uint8).flatten()
    UV = np.c_[U, V].flatten()
    yuv_ = np.c_[Y, UV].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422_UYVY(rgb_file, yuv_file):
    '''
    Y、U、V分量2:1:1，U、V在原来的基础上隔位采样
    YUV按照UYVY的组合排序
    内存排布：U0 Y0 V0 Y1 U1 Y2 V1 Y3 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::2], dtype=np.uint8).flatten()
    V = np.array(V[0::2], dtype=np.uint8).flatten()
    UV = np.c_[U, V].flatten()
    yuv_ = np.c_[UV, Y].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422_YVYU(rgb_file, yuv_file):
    '''
    Y、U、V分量2:1:1，U、V在原来的基础上隔位采样
    YUV按照YVYU的组合排序
    内存排布：Y0 V0 Y1 U0 Y2 V1 Y3 U1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::2], dtype=np.uint8).flatten()
    V = np.array(V[0::2], dtype=np.uint8).flatten()
    VU = np.c_[V, U].flatten()
    yuv_ = np.c_[Y, VU].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422_VYUY(rgb_file, yuv_file):
    '''
    Y、U、V分量2:1:1，U、V在原来的基础上隔位采样
    YUV按照VYUY的组合排序
    内存排布：V0 Y0 U0 Y1 V1 Y2 U1 Y3 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::2], dtype=np.uint8).flatten()
    V = np.array(V[0::2], dtype=np.uint8).flatten()
    VU = np.c_[V, U].flatten()
    yuv_ = np.c_[VU, Y].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv420p_I420(rgb_file, yuv_file):
    '''
    Y全量保留，U、V在原来的基础上每4位采样一次
    内存排布：Y0 Y1 Y2 ...U0 U1 U2 ...V0 V1 V2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::4], dtype=np.uint8).flatten()
    V = np.array(V[0::4], dtype=np.uint8).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(U)
        f.write(V)


def rgb_to_yuv420p_YV12(rgb_file, yuv_file):
    '''
    Y全量保留，U、V在原来的基础上每4位采样一次
    内存排布：Y0 Y1 Y2 ...V0 V1 V2 ...U0 U1 U2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::4], dtype=np.uint8).flatten()
    V = np.array(V[0::4], dtype=np.uint8).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(V)
        f.write(U)


def rgb_to_yuv420sp_NV12(rgb_file, yuv_file):
    '''
    Y全量保留，U、V在原来的基础上每4位采样一次
    先存放所有的Y，然后U、V交错存储
    内存排布：Y0 Y1 Y2 ...U0 V0 U1 V1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::4], dtype=np.uint8).flatten()
    V = np.array(V[0::4], dtype=np.uint8).flatten()
    UV = np.c_[U, V].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(UV)


def rgb_to_yuv420sp_NV21(rgb_file, yuv_file):
    '''
    Y全量保留，U、V在原来的基础上每4位采样一次
    先存放所有的Y，然后U、V交错存储
    内存排布：Y0 Y1 Y2 ...V0 U0 V1 U1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=np.uint8).flatten()
    U = np.array(yuv[:, :, 1], dtype=np.uint8).flatten()
    V = np.array(yuv[:, :, 2], dtype=np.uint8).flatten()
    U = np.array(U[0::4], dtype=np.uint8).flatten()
    V = np.array(V[0::4], dtype=np.uint8).flatten()
    VU = np.c_[V, U].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(VU)
