'''
调整图片到指定大小
'''

#coding=utf-8

import cv2
import sys

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'Usage: python resize.py src_file_name dst_file_name'
    exit(1)
  pic = cv2.imread(sys.argv[1])
  pic = cv2.resize(pic, (300, 300), interpolation=cv2.INTER_LINEAR)
  cv2.imwrite(sys.argv[2], pic)
