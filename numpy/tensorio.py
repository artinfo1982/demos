# -*- coding: UTF-8 -*-

import numpy as np


def load_tensor_from_text_file(file_name, dtype=np.float32, delimiter=','):
    return np.loadtxt(fname=file_name, dtype=dtype, delimiter=delimiter)


def load_tensor_from_binary_file(file_name):
    return np.fromfile(file_name)
