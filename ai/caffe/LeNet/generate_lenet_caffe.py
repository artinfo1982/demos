# -*- coding: utf-8 -*-

'''
使用pycaffe，生成LeNet基于caffe的网络定义文件
'''

import caffe
from caffe import layers as L
from caffe import params as P


def lenet(train_lmdb, test_lmdb, train_batch_size, test_batch_size, output_path):
    # train
    ntrain = caffe.NetSpec()
    # val
    nval = caffe.NetSpec()
    # deploy
    ndeploy = caffe.NetSpec()

    #--------------------------------------------------
    # train + val
    ntrain.data, ntrain.label = L.Data(name='data', batch_size=train_batch_size, backend=P.Data.LMDB,
                                       source=train_lmdb, transform_param=dict(scale=1. / 255), include=dict(phase=caffe.TRAIN), ntop=2)
    nval.data, nval.label = L.Data(name='data', batch_size=test_batch_size, backend=P.Data.LMDB,
                                   source=test_lmdb, transform_param=dict(scale=1. / 255), include=dict(phase=caffe.TEST), ntop=2)
    ntrain.conv1 = L.Convolution(ntrain.data, name='conv1', kernel_size=5, num_output=20, stride=1, weight_filler=dict(
        type='xavier'), bias_filler=dict(type='constant'), param=[dict(lr_mult=1), dict(lr_mult=2)])
    ntrain.pool1 = L.Pooling(ntrain.conv1, name='pool1', kernel_size=2,
                             stride=2, pool=P.Pooling.MAX)
    ntrain.conv2 = L.Convolution(ntrain.pool1, name='conv2', kernel_size=5, num_output=50, stride=1, weight_filler=dict(
        type='xavier'), bias_filler=dict(type='constant'), param=[dict(lr_mult=1), dict(lr_mult=2)])
    ntrain.pool2 = L.Pooling(ntrain.conv2, name='pool2', kernel_size=2,
                             stride=2, pool=P.Pooling.MAX)
    ntrain.ip1 = L.InnerProduct(ntrain.pool2, name='ip1', num_output=500,
                                weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'), param=[dict(lr_mult=1), dict(lr_mult=2)])
    ntrain.relu1 = L.ReLU(ntrain.ip1, name='relu1', in_place=True)
    ntrain.ip2 = L.InnerProduct(ntrain.relu1, name='ip2',
                                num_output=10, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'), param=[dict(lr_mult=1), dict(lr_mult=2)])
    ntrain.accuracy = L.Accuracy(
        ntrain.ip2, ntrain.label, name='accuracy', include=dict(phase=caffe.TEST))
    ntrain.loss = L.SoftmaxWithLoss(ntrain.ip2, ntrain.label, name='loss')

    #--------------------------------------------------
    # deploy，删去lr_mult、decay_mult、weight_filler、bias_filler
    # ({'shape': {'dim': [batch_size, channels, n_rows, n_cols]}})
    ndeploy.data = L.Input(input_param={'shape': {'dim': [64, 1, 28, 28]}})

    ndeploy.conv1 = L.Convolution(
        ndeploy.data, name='conv1', kernel_size=5, num_output=20, stride=1)
    ndeploy.pool1 = L.Pooling(ndeploy.conv1, name='pool1', kernel_size=2,
                              stride=2, pool=P.Pooling.MAX)
    ndeploy.conv2 = L.Convolution(
        ndeploy.pool1, name='conv2', kernel_size=5, num_output=50, stride=1)
    ndeploy.pool2 = L.Pooling(ndeploy.conv2, name='pool2', kernel_size=2,
                              stride=2, pool=P.Pooling.MAX)
    ndeploy.ip1 = L.InnerProduct(ndeploy.pool2, name='ip1', num_output=500)
    ndeploy.relu1 = L.ReLU(ndeploy.ip1, name='relu1', in_place=True)
    ndeploy.ip2 = L.InnerProduct(ndeploy.relu1, name='ip2',
                                 num_output=10)

    ndeploy.prob = L.Softmax(ndeploy.ip2, name='prob')

    out_train_test = str('name: "LeNet"\n') + \
        str(nval.to_proto()) + str(ntrain.to_proto())
    with open(output_path + '/lenet_train_val.prototxt', 'w') as f:
        f.write(out_train_test)

    out_deploy = str('name: "LeNet"\n') + str(ndeploy.to_proto())
    with open(output_path + '/lenet_deploy.prototxt', 'w') as f:
        f.write(out_deploy)


lenet('examples/mnist/mnist_train_lmdb', 'examples/mnist/mnist_test_lmdb',
      64, 100, '/home/zxh')
