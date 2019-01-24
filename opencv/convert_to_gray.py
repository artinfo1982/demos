'''
将彩色图片转为黑白图片
'''

import cv2
im_gray = cv2.imread('a.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('gray.jpg', im_gray)
