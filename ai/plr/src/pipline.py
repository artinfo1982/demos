# coding=utf-8

import detect
import cv2
import sys
import os
import hashlib
import finemapping as fm
import segmentation
import typeDistinguish as td
import finemapping_vertical as fv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
reload(sys)
sys.setdefaultencoding("utf-8")


def SimpleRecognizePlate(image):
    images = detect.detectPlateRough(
        image, image.shape[0], top_bottom_padding_rate=0.1)
    res_set = []
    for j, plate in enumerate(images):
        plate, rect, origin_plate = plate
        plate = cv2.resize(plate, (136, 36 * 2))
        ptype = td.SimplePredict(plate)
        if ptype > 0 and ptype < 5:
            plate = cv2.bitwise_not(plate)
        image_rgb = fm.findContoursAndDrawBoundingBox(plate)
        image_rgb = fv.finemappingVertical(image_rgb)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        val = segmentation.slidingWindowsEval(image_gray)
        if len(val) == 3:
            blocks, res, confidence = val
            if confidence > 0:
                print "车牌号码: %s" % res
            else:
                pass
    return image, res_set
