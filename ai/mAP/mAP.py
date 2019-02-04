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
        if ovmax >= 0.5 and not maxbbgt[5]:
            if not maxbbgt[4]:  # Ignore difficult objects
                tp[d] += 1
                maxbbgt[5] = True
        else:
            fp[d] += 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / cls_npos
    prec = tp / (fp + tp)

    if use_07_metric:
        ap = 0
        for t in np.linspace(0, 1, 11):
            patr = prec[rec >= t]
            p = 0 if len(patr) == 0 else max(patr)
            ap += p / 11
        return rec, prec, ap
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], rec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return mrec, mpre, ap


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='''A script of mAP calucation for object detection.''',
        epilog='Visit https://github.com/artinfo1982/demos/tree/master/ai/mAP for more details.'
    )
    parser.add_argument('-r', '--result',
                        type=str,
                        required - True,
                        metavar='DIR',
                        help='The directory contains detection results.'
                        )
    parser.add_argument('-t', '--ground-truth',
                        type=str,
                        required - True,
                        metavar='DIR',
                        help='The directory contains the ground truth.'
                        )
    parser.add_argument('-g', '--graph',
                        action='store_true',
                        help='Show the graphic result.'
                        )
    parser.add_argument('--use_07_metric',
                        action='store_true',
                        help='Calucate with VOC2007 metric if specified.'
                        )
    args = parser.parse_args(argv)

    gt, npos = load_ground_trunth(args.ground_truth)
    mAP = 0

    aps = []
    for cls in sorted(gt):
        _, _, ap = VOCevaldet(
            cls, gt[cls], npos[cls], args.result, args.use_07_metric
        )
        print(cls + ":", ap)
        mAP += ap / len(gt)
        aps.append(ap)
    print("--------------------------------------------------")
    print("mAP:", mAP)

    if args.graph:
        import matplotlib.pyplot as plt
        plt.rcdefaults()
        plt.barh(np.arange(len(gt)), list(reversed(aps)), align='center')
        plt.yticks(np.arange(len(gt)), sorted(gt.keys(), reverse=True))
        plt.xlabel('Average Precision')
        plt.xlim((0, 1.1))
        plt.title('mAP: %.1f%%' % (mAP * 100))
        for i in range(len(aps)):
            v = aps[-1 - i]
            vt = "%.3f" % v
            plt.text(v + .01, i, vt, color='blue', verticalalignment='center')
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
