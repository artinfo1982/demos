import numpy as np


def py_cpu_nms(dets, thresh):
    '''
    dets矩阵，是一个(Nx5)的二维矩阵，每一行分别表示为一个bbox对应的xmin，ymin，xmax，ymax，confidence(置信度)
    例如：
    dets = np.array([
        [204, 102, 358, 250, 0.7], # bbox1
        [257, 118, 380, 250, 0.2], # bbox2
        [280, 135, 400, 250, 0.6], # bbox3
        [255, 118, 360, 235, 0.4] # bbox4
        ...
    ])
    '''
    x1 = dets[:, 0]  # xmin，向量， 切第1列
    y1 = dets[:, 1]  # ymin，向量， 切第2列
    x2 = dets[:, 2]  # xmax，向量， 切第3列
    y2 = dets[:, 3]  # ymax，向量， 切第4列
    scores = dets[:, 4]  # confidence，向量，切第5列
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算每个bbox的面积，得到一个向量
    # numpy.argsort()，返回数组的下标索引，默认升序，如果np.argsort(-x)，表示降序排列x
    order = scores.argsort(-scores)

    keep = []  # 保存最终留下来的bbox
    while order.size > 0:  # 非空
        i = order[0]  # 取出当前order向量中第一个（置信度最大的那个）对应的下标索引
        keep.append(i)  # 将该索引存入keep
        '''
        以下代码计算每两个bbox之间的交叠面积
        ''''
        # 拿当前置信度最大的那一行的xmin和xmin向量的其余所有元素依次求max，得到一个子向量，N-1个元素
        xx1 = np.maximum(x1[i], x1[order[1:]])
        # 拿当前置信度最大的那一行的ymin和ymin向量的其余所有元素依次求max，得到一个子向量，N-1个元素
        yy1 = np.maximum(y1[i], y1[order[1:]])
        # 拿当前置信度最大的那一行的xmax和xmax向量的其余所有元素依次求min，得到一个子向量，N-1个元素
        xx2 = np.minimum(x2[i], x2[order[1:]])
        # 拿当前置信度最大的那一行的ymax和ymax向量的其余所有元素依次求min，得到一个子向量，N-1个元素
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 交叠矩形的宽，向量，表示两两bbox，N-1个元素
        w = np.maximum(0.0, xx2 - xx1 + 1)
        # 交叠矩形的高，向量，表示两两bbox，N-1个元素
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h  # 交叠矩形的面积，向量，表示两两bbox，N-1个元素
        # IOU计算公式=交叠面积/(并集面积-交叠面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # np.where(condition)，输出满足condition条件的内容，保留IOU小于等于阈值的bbox，删除IOU大于阈值的部分
        # np.where返回的是一个tuple，至少包含两个元素，例如(array([0, 1, 2], dtype=int64), ), 第一个元素是一个array，所以通过[0]获取
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]  # 将order数组的第一个元素剔除出去

    return keep
