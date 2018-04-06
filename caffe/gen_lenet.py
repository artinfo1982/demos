# -*- coding: utf-8 -*-

'''
使用pycaffe，生成LeNet基于caffe的网络定义文件
'''

import caffe
from caffe import layers as L
from caffe import params as P


def lenet(lmdb, batch_size, output_path):
    # train
    n = caffe.NetSpec()
    # test
    ntest = caffe.NetSpec()
    n.data, n.label = L.Data(name='mnist_train', batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), include=dict(phase=caffe.TRAIN), ntop=2)
    ntest.data, ntest.label = L.Data(name='mnist_test', batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                     transform_param=dict(scale=1. / 255), include=dict(phase=caffe.TEST), ntop=2)
    n.conv1 = L.Convolution(n.data, name='conv1', kernel_size=5,
                            num_output=20, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    n.pool1 = L.Pooling(n.conv1, name='pool1', kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, name='conv2', kernel_size=5, num_output=50, weight_filler=dict(
        type='xavier'), bias_filler=dict(type='constant'))
    n.pool2 = L.Pooling(n.conv2, name='pool2', kernel_size=2,
                        stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, name='ip1', num_output=500,
                           weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    n.relu1 = L.ReLU(n.ip1, name='relu1', in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, name='ip2',
                           num_output=10, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    n.soft = L.Softmax(n.ip2, name='soft')
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label, name='loss')
    n.acc = L.Accuracy(n.ip2, n.label, name='acc')

    out_train_test = str('name: "LeNet_mnist_train_test"') + \
        str(ntest.to_proto()) + str(n.to_proto())
    with open(output_path + '/lenet_mnist_train_test.prototxt', 'w') as f:
        f.write(out_train_test)

    out_deploy = str('name: "LeNet_mnist_deploy"') + str(n.to_proto())
    with open(output_path + '/lenet_mnist_deploy.prototxt', 'w') as f:
        f.write(out_deploy)


lenet('/home/zxh/caffe/examples/mnist/mnist_train_lmdb',
      64, '/home/zxh/caffe/test/mnist')
lenet('/home/zxh/caffe/examples/mnist/mnist_test_lmdb',
      100, '/home/zxh/caffe/test/mnist')
