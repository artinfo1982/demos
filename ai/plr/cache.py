import cv2
import os
import hashlib


def verticalMappingToFolder(image):
    name = hashlib.md5(image.data).hexdigest()[:8]
