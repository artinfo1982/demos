# -*- coding: utf-8 -*-

'''
演示selective search算法的使用。
selective search算法应用于RCNN、SPP、Fast-RCNN中，用于生成候选区域框
'''

from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import sys


def main():
    img = io.imread('cat.jpg')

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 1000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    sys.exit(main())
