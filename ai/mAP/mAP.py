import os
import sys
import argparse
import numpy as np
from collections import defaultdict


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def load_ground_trunth(gt_dir):
    gt = defaultdict(lambda: defaultdict(list))
    npos = defaultdict(int)
    for root, dirs, files in os.walk(gt_dir, topdown=False):
        for name in files:
            ids, _ = os.path.splitext(name)
            with open(os.path.join(root, name), "r") as f:
                for line in f:
                    cls, xmin, ymin, xmax, ymax, diff = line.split()
                    # The 6th element False represents whether the object
                    # has been detected.
                    gt[cls][ids].append(
                        [int(xmin), int(ymin), int(xmax),
                         int(ymax), int(diff) == 1, False]
                    )
                    npos[cls] += (1 - int(diff))
    return gt, npos


def VOCevaldet(cls, cls_gt, cls_npos, result_dir, use_07_metric=True):
    BB = np.empty((0, 4), dtype=np.int64)
    confidence = np.empty((0, 1), dtype=np.float64)
    ids = np.empty((0, 1), dtype=np.str)
    with open(os.path.join(result_dir, cls + ".txt"), "r") as f:
        for line in f:
            imid, conf, b1, b2, b3, b4 = line.split()
            ids = np.append(ids, imid)
            confidence = np.append(confidence, float(conf))
            BB = np.append(BB, [[int(b1), int(b2), int(b3), int(b4)]], axis=0)
    si = np.argsort(-confidence)
    ids = ids[si]
    BB = BB[si, :]
    nd = len(confidence)
    tp = np.zeros((nd, 1))
    fp = np.zeros((nd, 1))

    for d in range(nd):
        bb = BB[d, :]
        ovmax = -np.inf
        for bbgt in cls_gt[ids[d]]:
            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                  min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1
            ov = -np.inf
            if iw > 0 and ih > 0:
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                    (bbgt[2] - bbgt[0] + 1) * \
                    (bbgt[3] - bbgt[1] + 1) - (iw * ih)
                ov = iw * ih / ua
            if ov > ovmax:
                ovmax = ov
                maxbbgt = bbgt
