'''
Faster R-CNN
Faster R-CNN = RPN + Fast R-CNN（除去selective search外剩下的部分），bboxes通过rois换算来
Faster R-CNN的输入：
    data：经过缩放之后的ndarray格式的图片，float32类型
    im_info：图片信息，np.array，（高，宽，缩放因子），float32类型
Faster R-CNN的输出：
    cls_prob：分类置信度（每个类别的概率），cls_prob的shape=(N, 21)，其中N为roi的数目
    bbox_pred：边界框回归得到的边界框坐标修正值，取出来就是box_deltas
'''

import _init_paths
import matplotlib.pyplot as plt
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms
import numpy as np
import scipy.io as sio
import caffe
import os
import sys
import cv2
import time

MODE = 'CPU'
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])  # 论文中使用的像素均值
TEST_SCALES = (600,)  # 测试用的尺度，尺度是每个图片的最短的边长，论文中使用的值
TEST_MAX_SIZE = 1000  # 尺度化后，每个图片最长的边长，论文中使用的值
CONF_THRESH = 0.8
NMS_THRESH = 0.3


CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def im_list_to_blob(ims):
    '''
    把list中的ndarray格式的图片，转换为blob存储格式，要求图片都已经经过了预处理
    '''
    # 把所有图片的shape元组拼成一个矩阵(Nx3)，N表示图片的张数(batch)，取长宽数值最大的那一组shape，输出一维
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    # 预分配内存，shape=(N, max(heights), max(widths), 3)
    # numpy.zeros(shape, dtype=float, order='C')
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]  # i从0开始
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im  # 因为边长按照最大值，对于不足最大值的，还是0
    # 原始blob.shape=(N, max(heights), max(widths), 3)
    # 转置blob，新的shape=(N, 3, max(heights), max(widths))
    channel_swap = (0, 3, 1, 2)  # 重新定义axis顺序，原始4轴0, 1, 2, 3，新的轴0, 3, 1, 2
    # numpy.transpose(a, axes=None)，张量转置，输入新的轴顺序
    blob = blob.transpose(channel_swap)
    return blob


def _get_image_blob(im):
    """
    将一张普通图片按照多尺度缩放成若干图片（论文中偏向于单一尺度600）
    输入：
        im (ndarray)：BGR序的彩色图片
    输出：
        blob (ndarray): 保存图片金字塔信息的二进制数据结构
        im_scale_factors (list): 图片金字塔中使用的一组缩放因子
    """
    im_orig = im.astype(np.float32, copy=True)  # 复制出一个和原始图片一样的图片
    im_orig -= PIXEL_MEANS  # 减去像素均值
    im_shape = im_orig.shape  # shape一般是一个三元组，例如（300, 500, 3），3表示三通道
    im_size_min = np.min(im_shape[0:2])  # shape三元组中图片长宽的最小值，[m:n]表示从m到n-1
    im_size_max = np.max(im_shape[0:2])  # shape三元组中图片长宽的最大值，[m:n]表示从m到n-1
    processed_ims = []
    im_scale_factors = []
    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)  # 计算缩放比例
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:  # np.round，四舍五入
            # 重新调整缩放比例，确保按此比例放大，不会超过设定的最大值
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        # 调用opencv，按照缩放比例，重新调整图片大小，线性插值
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        # 针对每一种TEST_SCALES元组里的元素计算出的缩放比例因子，存进list
        im_scale_factors.append(im_scale)
        processed_ims.append(im)  # 处理过的图片，以ndarray的形式，存入list
    blob = im_list_to_blob(processed_ims)  # 构造blob数据结构，存放这些处理好的图片
    return blob, np.array(im_scale_factors)  # 返回blob和ndarray格式的缩放因子列表


def _get_blobs(im, rois):
    '''
    保存图片和其中的rois，如果使用rpn，则初始没有rois
    '''
    blobs = {'data': None, 'rois': None}  # 初始化一个字典
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors


def bbox_transform_inv(boxes, deltas):
    '''
    将boxes使用rpn网络产生的deltas进行坐标变换处理，求出变换后的boxes坐标，即预测的proposals。
    此处boxes一般表示原始rois，即未经过任何处理仅仅是经过平移之后产生的rois。
    输入：
        boxes：原始rois，二维，shape=(N, 4)，N表示rois的数目
        deltas：RPN网络产生的数据，二维，shape=(N, (1+classes)*4)，classes表示类别数目，1表示背景，N表示rois的数目
    输出：
        预测的变换之后的proposals（或者叫anchors）
    '''
    # boxes的shape=(N, 4)，其中4表示xmin、ymin、xmax、ymax，N为rois的数目
    if boxes.shape[0] == 0:  # rois为空
        # 返回一组0，换句话，不用调整任何box坐标
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)
    # xmax - xmin + 1，之所以加1，是防止xmin == xmax，widths shape=(N, 4)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    # ymax - ymin + 1，之所以加1，是防止ymin == ymax，heights shape=(N, 4)
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths  # 求得中心横坐标，xmin + w/2，ctr_x shape=(N, 4)
    # 求得中心纵坐标，ymin + h/2，ctr_y shape=(N, 4)
    ctr_y = boxes[:, 1] + 0.5 * heights
    # 获取每一个类别的deltas的信息，每一个类别的deltas的信息是顺序存储的，
    # 即第一个类别的四个信息（dx，dy，dw，dh）存储完成后才接着存另一个类别。
    # 下面四个变量的shape均为(N, classes+1)，N表示roi数目，classes表示类别数目(此处为20)，1表示背景
    # 双冒号的语法为：seq[start:end:step]，表示从start开始到end结束截取序列，步长为step。可以忽略end
    '''
    deltas shape=(N, (1+classes)*4)，存储结构示例：
    dx dy dw dh  dx dy dw dh  dx dy dw dh ...
    ---class 1---   ---class 2---   ---class 3---
                        ... N行
    deltas里面存放的都是缩放因子，也就是说dx，dy，dw，dh这些，都是缩放的系数
    dx，dy，是单纯的比例系数，直接和宽高相乘，就可以算出中心坐标
    dw，dh，是新的宽高相对于原来的宽高取对数log，因此，使用dw，dh时，需要使用exp
    '''
    dx = deltas[:, 0::4]  # 从第一个dx开始，间隔4抽取，得到所有的dx，dx shape=(N, 1+classes)
    dy = deltas[:, 1::4]  # 从第一个dy开始，间隔4抽取，得到所有的dy，dy shape=(N, 1+classes)
    dw = deltas[:, 2::4]  # 从第一个dw开始，间隔4抽取，得到所有的dw，dw shape=(N, 1+classes)
    dh = deltas[:, 3::4]  # 从第一个dh开始，间隔4抽取，得到所有的dh，dh shape=(N, 1+classes)
    '''
    np.newaxis，增加一个轴，一般用于扩增array、向量、矩阵，便于广播加或者广播乘
    a = np.array([[1, 2], [1, 2], [1, 2]]) # a的shape是(3, 2)
    b = np.array([2, 2, 2]) # b的shape是(3,)，注意，(3,)表示这是个array，并不是一个向量，a*b会报错，无法广播乘，此时就需要扩展b为向量
    array扩展为向量，有两种方案（行向量、列向量）
    b = b[np.newaxis, :] # 变成行向量，shape=(1, 3)
    b = b[:, np.newaxis] # 变成列向量，shape=(3, 1)
    这样，a*b，就可以广播乘了，(3, 2) * (3, 1) = (3, 2)
    a = [[a1, a2], [a3, a4], [a5, a6]]
    b = [[b1], [b2], [b3]]
    a */+ b = [[a1*/+b1, a2*/+b2], [a3*/+b1, a4*/+b2], [a5*/+b1, a6*/+b2]]
    '''
    # 原来的中心横坐标+宽度*缩放系数=新的中心横坐标
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    # 原来的中心纵坐标+高度*缩放系数=新的中心纵坐标
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # dw本来就已经取了对数，要还原为原始的缩放系数，必须使用exp
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    # dh本来就已经取了对数，要还原为原始的缩放系数，必须使用exp
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)  # 预分配空间，存放最终的预测框
    # pred_boxes的存储，类似deltas，不再赘述
    # x1，新的xmin
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1，新的ymin
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2，新的xmax
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2，新的ymax
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes


def clip_boxes(boxes, im_shape):
    '''
    如果预测框超出了原图的范围，调整预测框到原图内
    '''
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(
            np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for '
          '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = "/home/cd/py-faster-rcnn/models/pascal_voc/faster_rcnn_alt_opt/faster_rcnn_test.pt"
    caffemodel = "/home/cd/py-faster-rcnn/data/faster_rcnn_models/VGG16/VGG16_faster_rcnn_final.caffemodel"
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if MODE == 'GPU':
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
