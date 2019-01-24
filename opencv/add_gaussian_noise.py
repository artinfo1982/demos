'''
添加高斯噪声
'''

import cv2

img = cv2.imread('a.jpg')
#高斯滤波
img1 = cv2.GaussianBlur(img, (9, 9), 10)
cv2.imwrite('noise.jpg', img1)
