'''
In BGR order
'''

import os
import cv2
from numpy import *

img_dir = '/home/cd/TensorRT-4.0.1.6/data/resnet/jpgs'
img_list = os.listdir(img_dir)
sum_b = 0
sum_g = 0
sum_r = 0
b_mean = 0
g_mean = 0
r_mean = 0
count = 0

for img_name in img_list:
