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
  img_path = os.path.join(img_dir, img_name)
  img = cv2.imread(img_path)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # ro rgb
  sum_b = sum_b + img[:,:,0].mean()
  sum_g = sum_g + img[:,:,1].mean()
  sum_r = sum_r + img[:,:,2].mean()
  count = count + 1
  
b_mean = sum_b / count
g_mean = sum_g / count
r_mean = sum_r / count
img_mean = [b_mean, g_mean, r_mean]

print (img_mean)
