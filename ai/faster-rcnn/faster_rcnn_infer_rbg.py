# -*- coding: utf-8 -*-

'''
本程序改编自rgb大神的py-faster-rcnn，仅保留推理部分，代码详细注解，仅支持CPU模式
Faster R-CNN
Faster R-CNN = RPN + Fast R-CNN（除去selective search外剩下的部分），bboxes通过rois换算来
Faster R-CNN的输入：
    data：经过缩放之后的ndarray格式的图片，float32类型
    im_info：图片信息，np.array，（高，宽，缩放因子），float32类型
Faster R-CNN的输出：
    cls_prob：分类置信度（每个类别的概率），cls_prob的shape=(N, 21)，其中N为roi的数目
    bbox_pred：边界框回归得到的边界框坐标修正值，取出来就是box_deltas
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe
import os
import sys
import cv2
import time

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
    入参：
        im (ndarray)：BGR序的彩色图片
    返回：
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
    入参：
        boxes：原始rois，二维，shape=(N, 4)，N表示rois的数目
        deltas：RPN网络产生的数据，二维，shape=(N, (1+classes)*4)，classes表示类别数目，1表示背景，N表示rois的数目
    返回：
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
    # b的shape是(3,)，注意，(3,)表示这是个array，并不是一个向量，a*b会报错，无法广播乘，此时就需要扩展b为向量
    b = np.array([2, 2, 2])
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


def im_detect(net, im, boxes=None):
    '''
    根据给定的候选，识别图中的物体
    入参：
        net(caffe.Net)：此处使用Fast R-CNN网络
        im(ndarray)：BGR顺序的彩色图片
        boxes(ndarray)：RPN使用的Rx4的区域候选框
    返回：
        scores(ndarray)：物体类别的分数，RxK(K表示物体种类数，包含背景0)
        boxes(ndarray)：预测的边界框，Rx(4*K)
    '''
    blobs, im_scales = _get_blobs(im, boxes)  # 默认输入boxes为None
    # blobs中，key=data的value赋值给im_blob
    im_blob = blobs['data']
    # 参见函数 im_list_to_blob，转置后的blob顺序为 (N, 3, max(heights), max(widths))
    # im_blob.shape[2]和im_blob.shape[3]分别对应heights、width，
    # 再加上缩放因子，组成一个新的array，作为value，存入key='im_info'的blobs
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
    # python实现的caffe net，以有序字典blobs的形式保存了各层的信息
    # *号作用域元组，表示取出元组里的所有元素，例如t=(1, 2, 3)，*t=[1, 2, 3]
    # 把网络输入的data、im_info，调整为实际缩放后图片的大小
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    # 执行前向传播，即推理
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    # 两个星号，表示依次取map的key=value格式
    # 例如 forward_kwargs : {'data': xxx, 'im_info': xxx}
    # 下面这句就是net.forward(data=xxx, im_info=xxx)
    blobs_out = net.forward(**forward_kwargs)
    # 一个batch中只有一个图片，事实上确实是1，因为TEST_SCALES为一个元素的tuple
    assert len(im_scales) == 1
    rois = net.blobs['rois'].data.copy()  # 真正的rois数据从网络参数读入，随着给定的网络权值而定
    # 将网络参数中的rois，按照和真实图片的缩放因子缩放
    # rois的shape=(N, 4)，N是rois的数目，而boxes的shape同rois
    boxes = rois[:, 1:5] / im_scales[0]
    # 分类置信度（每个类别的概率），cls_prob的shape=(N, 21)，N是rois的数目
    # 注意，scores是一个简单的数组，array，并不是numpy的ndarray格式
    scores = blobs_out['cls_prob']
    # 执行边界框修正，即根据预测框偏差（box_deltas），经过坐标变换，调整原bbox的4个坐标
    box_deltas = blobs_out['bbox_pred']
    # 使用rpn网络产生的deltas，修正预测框的位置
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    # 如果预测框超出了原图的范围，调整预测框到原图内
    pred_boxes = clip_boxes(pred_boxes, im.shape)
    return scores, pred_boxes


def vis_detections(im, class_name, dets, ax, thresh=0.5):
    '''
    在一幅图中画出所有识别出的物体的边界框
    '''
    # dets[:, -1]，-1表示取右边倒数第一列，也就是confidence的那一列
    # np.where返回的是一个tuple，至少包含两个元素，例如(array([0, 1, 2], dtype=int64), )，第一个元素是一个array，所以通过[0]获取
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:  # 图中没有检测到任何物体
        return
    for i in inds:
        bbox = dets[i, :4]  # 取dets的第i行，第1-4列，4个坐标
        score = dets[i, -1]  # 取dets的第i行，第5列，confidence
        # matplotlib的axes.add_patch，设置属性
        # plt.Rectangle(xy, width, height, angle=0.0, **kwargs)
        #   xy：矩形左上角的x、y坐标，是一个二元tuple，(x, y)
        # bbox --- xmin、ymin、xmax、ymax
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0],
                                   bbox[3] - bbox[1], fill=False, edgecolor='red', linewidth=1))
        # 在图上添加文字，axes.text(x, y, s, fontdict=None, widthdash=False, **kwargs)
        ax.text(bbox[0], bbox[1] - 2, '%s %.3f' % (class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')
    # 设置图片的标题
    ax.set_title(('%s detections with p(%s | box) >= %.1f') %
                 (class_name, class_name, thresh), fontsize=14)


def demo(net, image_name):
    '''
    使用训练好的权值，检测图片
    '''
    # 载入图片，opencv读入之后，自动转为numpy的ndarray格式
    im = cv2.imread(image_name)
    # 检测图片中所有的物体种类，并计时
    begin_time = int(round(time.time() * 1000))  # 获取毫秒级当前时间，作为起始时间
    scores, boxes = im_detect(net, im)
    end_time = int(round(time.time() * 1000))  # 获取毫秒级当前时间，作为结束时间
    print('Detection took %d ms for %d object proposals' % (end_time - begin_time, boxes.shape[0])
    im=im[:, :, (2, 1, 0)]
    fig, ax=plt.subplot(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    # 为每一个识别出来的物体画边界框
    # python enumerate，返回元素的索引列表（从0开始）和元素本身的列表
    for cls_ind, cls in enumerate(CLASSES[1:]):  # CLASSES从下标1开始，已经去掉背景0
        cls_ind += 1  # 丢弃背景0
        # 依次取出所有的边界框的坐标，N组，每组4个
        # cls_boxes shape=(N, 4)
        cls_boxes=boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        # 依次取出所有的得分，N组，每组1个，cls_scores shape=(N,)，因为scores本身只是个简单的array
        cls_scores=scores[:, cls_ind]
        # np.hstack是水平拼接所有的数组、矩阵，np.vstack是垂直拼接
        # cls_scores[:, np.newaxis]之后，cls_scores shape=(N, 1)
        # hstack之后，dets shape=(N, 5)
        dets=np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep=py_cpu_nms(dets, NMS_THRESH)
        dets=dets[keep, :]  # 只保留nms之后留下的坐标+置信度元组
        # 将所有检测到的物体画边界框在一张图中
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
    plt.axis('off')
    # 紧凑显示图像
    plt.tight_layout()
    plt.draw()


if __name__ == '__main__':
    prototxt="/home/cd/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt"
    caffemodel="/home/cd/py-faster-rcnn/data/faster_rcnn_models/VGG16/VGG16_faster_rcnn_final.caffemodel"
    if not os.path.isfile(caffemodel):
        raise IOError('%s not found.\n' % caffemodel)

    caffe.set_mode_cpu()
    net=caffe.Net(prototxt, caffemodel, caffe.TEST)

    im_names=['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print('Demo for data/demo/%s' % im_name)
        demo(net, im_name)

    plt.show()
