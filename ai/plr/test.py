# -*- coding: utf-8 -*-

'''
车牌识别检测测试程序，可以识别检测车牌号码、车牌颜色
'''

from src import pipline as pp
from src import typeDistinguish as td
import cv2
import sys

inputFile = sys.argv[1]
image = cv2.imread(inputFile)
pp.SimpleRecognizePlate(image)
