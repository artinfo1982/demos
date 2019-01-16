'''
将jpg格式的图片，转为ppm
'''

#coding=utf-8

import sys
from PIL import Image

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'Usage: python jpg2ppm.py jpg_file_name ppm_file_name'
    exit(1)
  jpg_file_name = sys.argv[1]
  ppm_file_name = sys.argv[2]
  img = Image.open(jpg_file_name)
  img.save(ppm_file_name)
