# -*- coding: utf-8 -*-

'''
将RGB图片(例如JPG)转变为各种YUV

YUV是一种颜色编码的方法，因为人眼对于明暗的变化要比对彩色的变化敏感的多，因此可以对彩色进行降采样压缩编码。
Y(明亮度，Luminance)、U(色度，Chrominance)、V(浓度，Chroma)。
历史上，Y'UV、YCbCr、YPbPr等专有名词都可以称为YUV。

YUV和RGB的关系：
通过矩阵和向量的乘积得到，由此可见，原始YUV和RGB具有相同的维度和大小。
----------------------------------------------
Y     0.299       0.587      0.114           R
U = -0.14713    -0.28886     0.436           G
V     0.615     -0.51499    -0.10001         B
----------------------------------------------
R    1       0         1.13983          Y
G =  1    -0.39465    -0.58060          U
B    1    2.03211         0             V
----------------------------------------------

假设一张像素点为(M*N)的RGB图像，每个像素点有R、G、B三个分量。
经过转换，可以得到(M*N)大小的YUV点阵，每个点有Y、U、V三个分量。

YUV采样：
    4:0:0，没有降采样，只有Y分量，没有UV分量
    4:4:4，没有降采样，Y、U、V分量数目相同
    4:2:2，Y没有降采样，U、V分别水平方向1/2降采样，垂直方向不降采样，每2个Y共用一组UV分量
    4:2:0，Y没有降采样，U、V分别水平方向1/2降采样，垂直方向1/2降采样，每4个Y共用一组UV分量
    4:1:1，Y没有降采样，U、V分别水平方向1/4降采样，垂直方向不降采样，每4个Y共用一组UV分量

例如一个4x4像素的图片，忽略Y，仅分析UV。(X表示采样点，O表示不采样点)
对于4:2:2
        X       O       X       O
        X       O       X       O
        X       O       X       O
        X       O       X       O
对于4:2:0
        X       O      X      O
        O       O      O      O
        X       O      X      O
        O       O      O      O
对于4:1:1
        X       O       O       O
        X       O       O       O
        X       O       O       O
        X       O       O       O

    
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
yuv411p
yuv411sp
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


def rgb_to_yuv400(rgb_file, yuv_file, dtype=np.uint8):
    '''
    只有Y分量
        +--------------+
        |Y0Y1Y2...     |
        |...           | h
        +--------------+
               w
    内存排布：Y0 Y1 Y2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype)
    with open(yuv_file, 'wb') as f:
        f.write(Y)


def rgb_to_yuv444p(rgb_file, yuv_file, dtype=np.uint8):
    '''
        +--------------+
        |Y0Y1Y2...     |
        |...           |
        |U0U1U2...     |
        |...           | h
        |V0V1V2...     |
        |...           |
        +--------------+
               w
    内存排布：Y0 Y1 Y2 ...U0 U1 U2 ...V0 V1 V2 ...
    '''
    yuv = np.array(rgb2yuv(rgb_file), dtype=dtype)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(U)
        f.write(V)


def rgb_to_yuv444(rgb_file, yuv_file, dtype=np.uint8):
    '''
        +-----------------------+
        |Y0U0V0Y1U1V1...        |
        |...                    | h
        +-----------------------+
                  w
    内存排布：Y0 U0 V0 ...Y1 U1 V1 ...Y2 U2 V2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    yuv_ = np.c_[Y, U, V].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422p(rgb_file, yuv_file, dtype=np.uint8):
    '''
                      w
            +--------------------+
            |Y0Y1Y2Y3...         |
            |...                 | h
            +--------------------+
            |U0U1...   |
            |...       | h
            +----------+
            |V0V1...   |
            |...       | h
            +----------+
                w/2
    内存排布：Y0 Y1 Y2 ...U0 U1 U2 ...V0 V1 V2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::2], dtype=dtype).flatten()
    V = np.array(V[0::2], dtype=dtype).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(U)
        f.write(V)


def rgb_to_yuv422sp(rgb_file, yuv_file, dtype=np.uint8):
    '''
                         w
            +---------------------------+
            |Y0Y1Y2Y3...                |
            |...                        | h
            +---------------------------+
            |U0V0U1V1... |
            |...         | h
            +------------+
                  w/2
    内存排布：Y0 Y1 Y2 ...U0 V0 U1 V1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::2], dtype=dtype).flatten()
    V = np.array(V[0::2], dtype=dtype).flatten()
    UV = np.c_[U, V].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(UV)


def rgb_to_yuv422_YUYV(rgb_file, yuv_file, dtype=np.uint8):
    '''
                         2w
            +---------------------------+
            |Y0U0Y1V0Y2U1Y3V1...        |
            |...                        | h
            +---------------------------+
    内存排布：Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::2], dtype=dtype).flatten()
    V = np.array(V[0::2], dtype=dtype).flatten()
    UV = np.c_[U, V].flatten()
    yuv_ = np.c_[Y, UV].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422_UYVY(rgb_file, yuv_file, dtype=np.uint8):
    '''
                          2w
            +---------------------------+
            |U0Y0V0Y1U1Y2V1Y3...        |
            |...                        | h
            +---------------------------+
    内存排布：U0 Y0 V0 Y1 U1 Y2 V1 Y3 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::2], dtype=dtype).flatten()
    V = np.array(V[0::2], dtype=dtype).flatten()
    UV = np.c_[U, V].flatten()
    yuv_ = np.c_[UV, Y].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422_YVYU(rgb_file, yuv_file, dtype=np.uint8):
    '''
                         2w
            +---------------------------+
            |Y0V0Y1U0Y2V1Y3U1...        |
            |...                        | h
            +---------------------------+
    内存排布：Y0 V0 Y1 U0 Y2 V1 Y3 U1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::2], dtype=dtype).flatten()
    V = np.array(V[0::2], dtype=dtype).flatten()
    VU = np.c_[V, U].flatten()
    yuv_ = np.c_[Y, VU].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv422_VYUY(rgb_file, yuv_file, dtype=np.uint8):
    '''
                         2w
            +---------------------------+
            |V0Y0U0Y1V1Y2U1Y3...        |
            |...                        | h
            +---------------------------+
    内存排布：V0 Y0 U0 Y1 V1 Y2 U1 Y3 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::2], dtype=dtype).flatten()
    V = np.array(V[0::2], dtype=dtype).flatten()
    VU = np.c_[V, U].flatten()
    yuv_ = np.c_[VU, Y].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(yuv_)


def rgb_to_yuv420p_I420(rgb_file, yuv_file, height, width, dtype=np.uint8):
    '''
                      w
            +--------------------+
            |Y0Y1Y2Y3...         |
            |...                 | h
            +--------------------+
            |U0U1...   | h/2
            +----------+
            |V0V1...   | h/2
            +----------+
                 w/2
    内存排布：Y0 Y1 Y2 ...U0 U1 U2 ...V0 V1 V2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).reshape(height, width)
    V = np.array(yuv[:, :, 2], dtype=dtype).reshape(height, width)
    # 水平和垂直方向都1/2降采样
    U = np.array(U[0::2, 0::2], dtype=dtype).flatten()
    V = np.array(V[0::2, 0::2], dtype=dtype).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(U)
        f.write(V)


def rgb_to_yuv420p_YV12(rgb_file, yuv_file, height, width, dtype=np.uint8):
    '''
                      w
            +--------------------+
            |Y0Y1Y2Y3...         |
            |...                 | h
            +--------------------+
            |V0V1...   | h/2
            +----------+
            |U0U1...   | h/2
            +----------+
                 w/2
    内存排布：Y0 Y1 Y2 ...V0 V1 V2 ...U0 U1 U2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).reshape(height, width)
    V = np.array(yuv[:, :, 2], dtype=dtype).reshape(height, width)
    # 水平和垂直方向都1/2降采样
    U = np.array(U[0::2, 0::2], dtype=dtype).flatten()
    V = np.array(V[0::2, 0::2], dtype=dtype).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(V)
        f.write(U)


def rgb_to_yuv420sp_NV12(rgb_file, yuv_file, height, width, dtype=np.uint8):
    '''
                         w
            +---------------------------+
            |Y0Y1Y2Y3...                |
            |...                        | h
            +---------------------------+
            |U0V0U1V1...  | h/2
            +-------------+
                  w/2
    内存排布：Y0 Y1 Y2 ...U0 V0 U1 V1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).reshape(height, width)
    V = np.array(yuv[:, :, 2], dtype=dtype).reshape(height, width)
    # 水平和垂直方向都1/2降采样
    U = np.array(U[0::2, 0::2], dtype=dtype).flatten()
    V = np.array(V[0::2, 0::2], dtype=dtype).flatten()
    UV = np.c_[U, V].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(UV)


def rgb_to_yuv420sp_NV21(rgb_file, yuv_file, height, width, dtype=np.uint8):
    '''
                         w
            +---------------------------+
            |Y0Y1Y2Y3...                |
            |...                        | h
            +---------------------------+
            |V0U0V1U1...  | h/2
            +-------------+
                  w/2
    内存排布：Y0 Y1 Y2 ...V0 U0 V1 U1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).reshape(height, width)
    V = np.array(yuv[:, :, 2], dtype=dtype).reshape(height, width)
    # 水平和垂直方向都1/2降采样
    U = np.array(U[0::2, 0::2], dtype=dtype).flatten()
    V = np.array(V[0::2, 0::2], dtype=dtype).flatten()
    VU = np.c_[V, U].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(VU)


def rgb_to_yuv411p(rgb_file, yuv_file, dtype=np.uint8):
    '''
                      w
            +--------------------+
            |Y0Y1Y2Y3...         |
            |...                 | h
            +--------------------+
            |U0U1...   | h/2
            +----------+
            |V0V1...   | h/2
            +----------+
                w/2
    内存排布：Y0 Y1 Y2 ...U0 U1 U2 ...V0 V1 V2 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::4], dtype=dtype).flatten()
    V = np.array(V[0::4], dtype=dtype).flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(U)
        f.write(V)


def rgb_to_yuv411sp(rgb_file, yuv_file, dtype=np.uint8):
    '''
                         w
            +---------------------------+
            |Y0Y1Y2Y3...                |
            |...                        | h
            +---------------------------+
            |U0V0U1V1...                | h/2
            +---------------------------+
                         w
    内存排布：Y0 Y1 Y2 ...U0 V0 U1 V1 ...
    '''
    yuv = rgb2yuv(rgb_file)
    Y = np.array(yuv[:, :, 0], dtype=dtype).flatten()
    U = np.array(yuv[:, :, 1], dtype=dtype).flatten()
    V = np.array(yuv[:, :, 2], dtype=dtype).flatten()
    U = np.array(U[0::4], dtype=dtype).flatten()
    V = np.array(V[0::4], dtype=dtype).flatten()
    UV = np.c_[U, V].flatten()
    with open(yuv_file, 'wb') as f:
        f.write(Y)
        f.write(UV)
