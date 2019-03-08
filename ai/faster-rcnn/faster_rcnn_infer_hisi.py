# -*- coding: utf-8 -*-

import struct
import numpy as np
import sys
from collections import defaultdict

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
result = defaultdict(list)


def load_tensor_from_custom_packed_binary_file(file_path, dtype=np.float32):
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            rank = int.from_bytes(struct.unpack_from(
                'I', content), byteorder='big')
            shape = struct.unpack_from('%dI' % rank, content, 4)
            tensor = np.frombuffer(
                content, dtype=dtype, count=-1, offset=4 + rank * 4).reshape(shape)
            return tensor
    except IOError as e:
        print(str(e))


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return keep


def vis_detections(cls_ind, class_name, dets, thresh=0.5, image_id=None):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        result[class_name].append(
            (image_id, score, bbox[0], bbox[1], bbox[2], bbox[3]))


def clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def main(argv=None):
    input_ = load_tensor_from_custom_packed_binary_file('face.t')
    rois = load_tensor_from_custom_packed_binary_file('a.t').reshape(304, 5)
    boxes = rois[:, 1:5]
    scores = load_tensor_from_custom_packed_binary_file('b.t').reshape(304, 11)
    deltas = load_tensor_from_custom_packed_binary_file('c.t').reshape(304, 44)
    pred_boxes = bbox_transform_inv(boxes, deltas)
    boxes = clip_boxes(pred_boxes,
                       (int(input_.shape[2]/1),
                        (input_.shape[3] / 1))
                       )
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(cls_ind, cls, dets, thresh=CONF_THRESH, image_id='face')
    for cls, pred in result.items():
        print(cls)
        print(pred)


if __name__ == '__main__':
    sys.exit(main())
