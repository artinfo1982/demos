import numpy as np


def iou(rectangle1, rectangle2):
    '''
    computing IoU
    IoU = intersection area / (union area - intersection area)
    param: rectangle1, rectangle2 (xmin, ymin, xmax, ymax)
    return: iou ratio
    '''
    # no intersection
    if rectangle1[2] <= rectangle2[0] or rectangle1[0] >= rectangle2[2]:
        return 0
    if rectangle1[3] <= rectangle2[1] or rectangle1[1] >= rectangle2[3]:
        return 0

    # intersection area
    sorted_x = np.sort(
        np.array([rectangle1[0], rectangle1[2], rectangle2[0], rectangle2[2]]))
    sorted_y = np.sort(
        np.array([rectangle1[1], rectangle1[3], rectangle2[1], rectangle2[3]]))
    i_area = (sorted_x[2] - sorted_x[1]) * (sorted_y[2] - sorted_y[1])

    # union area
    area1 = np.fabs(rectangle1[2] - rectangle1[0]) * \
        np.fabs(rectangle1[3] - rectangle1[1])
    area2 = np.fabs(rectangle2[2] - rectangle2[0]) * \
        np.fabs(rectangle2[3] - rectangle2[1])
    u_area = area1 + area2 - i_area
    return i_area / u_area
