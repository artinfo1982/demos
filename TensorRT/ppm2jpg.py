'''
将ppm格式的图片，转为jpg
'''

#coding=utf-8

import sys
from PIL import Image

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'Usage: python ppm2jpg.py ppm_file_name jpg_file_name'
    exit(1)
  ppm_file_name = sys.argv[1]
  jpg_file_name = sys.argv[2]
  img = Image.open(ppm_file_name)
  img.save(jpg_file_name)
