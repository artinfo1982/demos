# -*- coding: utf-8 -*-

'''
使用pycaffe，生成ResNet基于caffe的网络定义文件
'''

import caffe
from caffe import layers as L
from caffe import params as P


def conv(bottom, name, ks, stride, nout, pad):
    return L.Convolution(bottom, name=name, kernel_size=ks, stride=stride, num_output=nout,
                         pad=pad, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))


def bn(bottom, name, isUseGlobalStats):
    return L.BatchNorm(bottom, name=name, in_place=True, batch_norm_param=dict(
        use_global_stats=isUseGlobalStats))


def scale(bottom, name):
    return L.Scale(bottom, name=name, in_place=True,
                   scale_param=dict(bias_term=True))


def relu(bottom, name):
    return L.ReLU(bottom, name=name, in_place=True)


def pool_max(bottom, name, ks, stride):
    return L.Pooling(bottom, name=name, kernel_size=ks,
                     stride=stride, pool=P.Pooling.MAX)


def pool_ave(bottom, name, ks, stride):
    return L.Pooling(bottom, name=name, kernel_size=ks,
                     stride=stride, pool=P.Pooling.AVE)


def eltwise(left, right, name):
    return L.Eltwise(left, right, name=name, operation=P.Eltwise.SUM)


def innerProduct(bottom, nout):
    return L.InnerProduct(bottom, num_output=nout,
                          weight_filler=dict(type='xavier'),
                          bias_filler=dict(type='constant'))


def resnet18(train_lmdb, test_lmdb, train_batch_size, test_batch_size, output_path):
    # train
    n = caffe.NetSpec()
    # test
    ntest = caffe.NetSpec()
    n.data, n.label = L.Data(name='mnist_train', batch_size=train_batch_size, backend=P.Data.LMDB,
                             source=train_lmdb, transform_param=dict(scale=1. / 255), include=dict(phase=caffe.TRAIN), ntop=2)
    ntest.data, ntest.label = L.Data(name='mnist_test', batch_size=test_batch_size, backend=P.Data.LMDB,
                                     source=test_lmdb, transform_param=dict(scale=1. / 255), include=dict(phase=caffe.TEST), ntop=2)

    # 1
    n._1_conv = conv(n.data, '1_conv', 7, 2, 64, 3)
    n._1_bn = bn(n._1_conv, '1_bn', False)
    n._1_scale = scale(n._1_bn, '1_scale')
    n._1_relu = relu(n._1_scale, '1_relu')

    # 2
    n._2_pool = pool_max(n._1_relu, '2_pool', 3, 2)
    # a
    n._2_0_conv = conv(n._2_pool, '2_0_conv', 3, 1, 64, 1)
    n._2_0_bn = bn(n._2_0_conv, '2_0_bn', False)
    n._2_0_scale = scale(n._2_0_bn, '2_0_scale')

    n._2_1_1_conv = conv(n._2_pool, '2_1_1_conv', 3, 1, 64, 1)
    n._2_1_1_bn = bn(n._2_1_1_conv, '2_1_1_bn', False)
    n._2_1_1_scale = scale(n._2_1_1_bn, '2_1_1_scale')
    n._2_1_1_relu = relu(n._2_1_1_scale, '2_1_1_relu')

    n._2_1_2_conv = conv(n._2_1_1_relu, '2_1_2_conv', 3, 1, 64, 1)
    n._2_1_2_bn = bn(n._2_1_2_conv, '2_1_2_bn', False)
    n._2_1_2_scale = scale(n._2_1_2_bn, '2_1_2_scale')

    n._2_1_wise = eltwise(n._2_0_scale, n._2_1_2_scale, '2_1_eltwise')
    n._2_1_relu = relu(n._2_1_wise, '2_1_relu')
    # b
    n._2_2_1_conv = conv(n._2_1_relu, '2_2_1_conv', 3, 1, 64, 1)
    n._2_2_1_bn = bn(n._2_2_1_conv, '2_2_1_bn', False)
    n._2_2_1_scale = scale(n._2_2_1_bn, '2_2_1_scale')
    n._2_2_1_relu = relu(n._2_2_1_scale, '2_2_1_relu')

    n._2_2_2_conv = conv(n._2_2_1_relu, '2_2_2_conv', 3, 1, 64, 1)
    n._2_2_2_bn = bn(n._2_2_2_conv, '2_2_2_bn', False)
    n._2_2_2_scale = scale(n._2_2_2_bn, '2_2_2_scale')

    n._2_2_wise = eltwise(n._2_1_relu, n._2_2_2_scale, '2_2_eltwise')
    n._2_2_relu = relu(n._2_2_wise, '2_2_relu')

    # 3
    # a
    n._3_0_conv = conv(n._2_2_relu, '3_0_conv', 3, 1, 128, 1)
    n._3_0_bn = bn(n._3_0_conv, '3_0_bn', False)
    n._3_0_scale = scale(n._3_0_bn, '3_0_scale')

    n._3_1_1_conv = conv(n._2_2_relu, '3_1_1_conv', 3, 1, 128, 1)
    n._3_1_1_bn = bn(n._3_1_1_conv, '3_1_1_bn', False)
    n._3_1_1_scale = scale(n._3_1_1_bn, '3_1_1_scale')
    n._3_1_1_relu = relu(n._3_1_1_scale, '3_1_1_relu')

    n._3_1_2_conv = conv(n._3_1_1_relu, '3_1_2_conv', 3, 1, 128, 1)
    n._3_1_2_bn = bn(n._3_1_2_conv, '3_1_2_bn', False)
    n._3_1_2_scale = scale(n._3_1_2_bn, '3_1_2_scale')

    n._3_1_wise = eltwise(n._3_0_scale, n._3_1_2_scale, '3_1_eltwise')
    n._3_1_relu = relu(n._3_1_wise, '3_1_relu')
    # b
    n._3_2_1_conv = conv(n._3_1_relu, '3_2_1_conv', 3, 1, 128, 1)
    n._3_2_1_bn = bn(n._3_2_1_conv, '3_2_1_bn', False)
    n._3_2_1_scale = scale(n._3_2_1_bn, '3_2_1_scale')
    n._3_2_1_relu = relu(n._3_2_1_scale, '3_2_1_relu')

    n._3_2_2_conv = conv(n._3_2_1_relu, '3_2_2_conv', 3, 1, 128, 1)
    n._3_2_2_bn = bn(n._3_2_2_conv, '3_2_2_bn', False)
    n._3_2_2_scale = scale(n._3_2_2_bn, '3_2_2_scale')

    n._3_2_wise = eltwise(n._3_1_relu, n._3_2_2_scale, '3_2_eltwise')
    n._3_2_relu = relu(n._3_2_wise, '3_2_relu')

    # 4
    # a
    n._4_0_conv = conv(n._3_2_relu, '4_0_conv', 3, 1, 256, 1)
    n._4_0_bn = bn(n._4_0_conv, '4_0_bn', False)
    n._4_0_scale = scale(n._4_0_bn, '4_0_scale')

    n._4_1_1_conv = conv(n._3_2_relu, '4_1_1_conv', 3, 1, 256, 1)
    n._4_1_1_bn = bn(n._4_1_1_conv, '4_1_1_bn', False)
    n._4_1_1_scale = scale(n._4_1_1_bn, '4_1_1_scale')
    n._4_1_1_relu = relu(n._4_1_1_scale, '4_1_1_relu')

    n._4_1_2_conv = conv(n._4_1_1_relu, '4_1_2_conv', 3, 1, 256, 1)
    n._4_1_2_bn = bn(n._4_1_2_conv, '4_1_2_bn', False)
    n._4_1_2_scale = scale(n._4_1_2_bn, '4_1_2_scale')

    n._4_1_wise = eltwise(n._4_0_scale, n._4_1_2_scale, '4_1_eltwise')
    n._4_1_relu = relu(n._4_1_wise, '4_1_relu')
    # b
    n._4_2_1_conv = conv(n._4_1_relu, '4_2_1_conv', 3, 1, 256, 1)
    n._4_2_1_bn = bn(n._4_2_1_conv, '4_2_1_bn', False)
    n._4_2_1_scale = scale(n._4_2_1_bn, '4_2_1_scale')
    n._4_2_1_relu = relu(n._4_2_1_scale, '4_2_1_relu')

    n._4_2_2_conv = conv(n._4_2_1_relu, '4_2_2_conv', 3, 1, 256, 1)
    n._4_2_2_bn = bn(n._4_2_2_conv, '4_2_2_bn', False)
    n._4_2_2_scale = scale(n._4_2_2_bn, '4_2_2_scale')

    n._4_2_wise = eltwise(n._4_1_relu, n._4_2_2_scale, '4_2_eltwise')
    n._4_2_relu = relu(n._4_2_wise, '4_2_relu')

    # 5
    # a
    n._5_0_conv = conv(n._4_2_relu, '5_0_conv', 3, 1, 512, 1)
    n._5_0_bn = bn(n._5_0_conv, '5_0_bn', False)
    n._5_0_scale = scale(n._5_0_bn, '5_0_scale')

    n._5_1_1_conv = conv(n._4_2_relu, '5_1_1_conv', 3, 1, 512, 1)
    n._5_1_1_bn = bn(n._5_1_1_conv, '5_1_1_bn', False)
    n._5_1_1_scale = scale(n._5_1_1_bn, '5_1_1_scale')
    n._5_1_1_relu = relu(n._5_1_1_scale, '5_1_1_relu')

    n._5_1_2_conv = conv(n._5_1_1_relu, '5_1_2_conv', 3, 1, 512, 1)
    n._5_1_2_bn = bn(n._5_1_2_conv, '5_1_2_bn', False)
    n._5_1_2_scale = scale(n._5_1_2_bn, '5_1_2_scale')

    n._5_1_wise = eltwise(n._5_0_scale, n._5_1_2_scale, '5_1_eltwise')
    n._5_1_relu = relu(n._5_1_wise, '5_1_relu')
    # b
    n._5_2_1_conv = conv(n._5_1_relu, '5_2_1_conv', 3, 1, 512, 1)
    n._5_2_1_bn = bn(n._5_2_1_conv, '5_2_1_bn', False)
    n._5_2_1_scale = scale(n._5_2_1_bn, '5_2_1_scale')
    n._5_2_1_relu = relu(n._5_2_1_scale, '5_2_1_relu')

    n._5_2_2_conv = conv(n._5_2_1_relu, '5_2_2_conv', 3, 1, 512, 1)
    n._5_2_2_bn = bn(n._5_2_2_conv, '5_2_2_bn', False)
    n._5_2_2_scale = scale(n._5_2_2_bn, '5_2_2_scale')

    n._5_2_wise = eltwise(n._5_1_relu, n._5_2_2_scale, '5_2_eltwise')
    n._5_2_relu = relu(n._5_2_wise, '5_2_relu')

    n._5_pool = pool_ave(n._5_2_relu, '5_pool', 7, 1)
    n.ip = innerProduct(n._5_pool, 1000)
    n.soft = L.Softmax(n.ip, name='soft')
    n.loss = L.SoftmaxWithLoss(n.ip, n.label, name='loss')
    n.acc = L.Accuracy(n.ip, n.label, name='acc')

    out_train_test = str('name: "ResNet18_mnist_train_test"\n') + \
        str(ntest.to_proto()) + str(n.to_proto())
    with open(output_path + '/resnet18_mnist_train_test.prototxt', 'w') as f:
        f.write(out_train_test)

    out_deploy = str('name: "ResNet18_mnist_deploy"\n') + str(n.to_proto())
    with open(output_path + '/resnet18_mnist_deploy.prototxt', 'w') as f:
        f.write(out_deploy)


resnet18('/home/zxh/caffe/examples/mnist/mnist_train_lmdb', '/home/zxh/caffe/examples/mnist/mnist_test_lmdb',
         64, 100, '/home/zxh/caffe/test/resnet')
