# -*- coding: UTF-8 -*-

'''
struct can be used to resolve customized binary data, supported format is below:

x(padding) --- 1 byte
c(char) --- 1 byte
b(signed char) --- 1 byte
B(unsigned char) --- 1 byte
?(bool) --- 1 byte
h(short) --- 2 bytes
H(unsigned short) --- 2 bytes
i(int) --- 4 bytes
I(unsigned int) --- 4 bytes
l(long) --- 4 bytes
L(unsigned long) --- 4 bytes
q(long long) --- 8 bytes
Q(unsigned long long) --- 8 bytes
f(float) --- 4 bytes
d(double) --- 8 bytes
s(char[]) --- 1 byte
p(char[]) --- 1 byte
P(void*) --- depends on arch

unpack e.g.
struct Header
{
    unsigned short id;
    char[4] tag;
    unsigned int version;
    unsigned int count;
}
import struct
id, tag, version, count = struct.unpack("H4s2I", s)

pack e.g.
ss = struct.pack("H4s2I", id, tag, version, count)
'''

import numpy as np
import struct


def load_tensor_from_text_file_directly(file_path, dtype=np.float32, delimiter=','):
    try:
        # read data from text file, it's shape is arrangement in this file
        return np.loadtxt(fname=file_path, dtype=dtype, delimiter=delimiter)
    except IOError as e:
        print(str(e))


def load_tensor_from_text_file_with_reshape(file_path, shape, dtype=np.float32, delimiter=','):
    try:
        # original, numpy read data from text file, it's shape is arrangment in this file
        # we can change the shape
        return np.loadtxt(fname=file_path, dtype=dtype, delimiter=delimiter).reshape(shape)
    except IOError as e:
        print(str(e))


def load_tensor_from_binary_file_directly(file_path):
    try:
        # read data from binary file, data is flatten, means N x 1
        return np.fromfile(file=file_path)
    except IOError as e:
        print(str(e))


def load_tensor_from_custom_packed_binary_file(file_path, dtype=np.float32):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            # struct.unpack_from(fmt, buffer[, offset=0]), return a tuple
            # e.g. 04 00 00 00 01 00 00 00  67 00 00 00 01 00 00 00 01 00 00 00
            # rank = 04 00 00 00 = 4, byteorder is big
            # shape = 01 00 00 00 67 00 00 00 01 00 00 00 01 00 00 00 = (1, 103, 1, 1)
            rank = int.from_bytes(struct.unpack_from(
                'I', content), byteorder='big')
            # shape is a tuple with 4 int, that is 4*4 = 16 bytes
            shape = struct.unpack_from('%dI' % rank, content, 4)
            # numpy.frombuffer(buffer, dtype=float, count=-1, offset=0), begin after shape tuple
            # original, data in buffer is flatten, we must reorganize them according to shape
            # return a tensor, which can be used in tensorflow
            tensor = np.frombuffer(
                content, dtype=dtype, count=-1, offset=4 + rank * 4).reshape(shape)
            return tensor
    except IOError as e:
        print(str(e))


def write_tensor_to_binary_file(tensor, file_path):
    # rank is int type
    rank = tensor.ndim
    # tensor.shape is a tuple, can not combine a int with a tuple
    # correct way is combine a tuple with a tuple, so we must change int to tuple, just use (int, )
    # t is a tuple
    t = (rank, ) + tensor.shape
    try:
        with open(file_path, 'wb') as f:
            # *tuple means unpack a tuple
            # e.g. tuple=(1,2,3,4), *tuple is 1, 2, 3, 4, one item to 4 items
            f.write(struct.pack('I%dI' % rank, *t))
            f.write(tensor.tobytes())
    except IOError as e:
        print(str(e))
