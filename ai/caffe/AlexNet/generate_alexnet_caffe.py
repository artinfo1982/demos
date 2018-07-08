# -*- coding: utf-8 -*-

'''
使用pycaffe，生成AlexNet基于caffe的网络定义文件
'''

import caffe
from caffe import layers as L
from caffe import params as P


def alexnet(train_lmdb, test_lmdb, mean_file, train_batch_size, test_batch_size, output_path):
    # train
    ntrain = caffe.NetSpec()
    # val
    nval = caffe.NetSpec()
    # deploy
    ndeploy = caffe.NetSpec()

    #--------------------------------------------------
    # train + val
    ntrain.data, ntrain.label = L.Data(name='data', batch_size=train_batch_size, backend=P.Data.LMDB,
                                       source=train_lmdb, transform_param=dict(mirror=True, crop_size=227, mean_file=mean_file), include=dict(phase=caffe.TRAIN), ntop=2)
    nval.data, nval.label = L.Data(name='data', batch_size=test_batch_size, backend=P.Data.LMDB,
                                   source=test_lmdb, transform_param=dict(mirror=False, crop_size=227, mean_file=mean_file), include=dict(phase=caffe.TEST), ntop=2)

    ntrain.conv1 = L.Convolution(ntrain.data, name='conv1', kernel_size=11, num_output=96, stride=4, weight_filler=dict(
        type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ntrain.relu1 = L.ReLU(ntrain.conv1, name='relu1', in_place=True)
    ntrain.norm1 = L.LRN(ntrain.relu1, name='norm1',
                         local_size=5, alpha=1e-4, beta=0.75)
    ntrain.pool1 = L.Pooling(ntrain.norm1, name='pool1', kernel_size=3,
                             stride=2, pool=P.Pooling.MAX)

    ntrain.conv2 = L.Convolution(ntrain.pool1, name='conv2', kernel_size=5, num_output=256, pad=2, group=2, weight_filler=dict(
        type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.1), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ntrain.relu2 = L.ReLU(ntrain.conv2, name='relu2', in_place=True)
    ntrain.norm2 = L.LRN(ntrain.relu2, name='norm2',
                         local_size=5, alpha=1e-4, beta=0.75)
    ntrain.pool2 = L.Pooling(ntrain.norm2, name='pool2', kernel_size=3,
                             stride=2, pool=P.Pooling.MAX)

    ntrain.conv3 = L.Convolution(ntrain.pool2, name='conv3', kernel_size=3, num_output=384, pad=1, weight_filler=dict(
        type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ntrain.relu3 = L.ReLU(ntrain.conv3, name='relu3', in_place=True)

    ntrain.conv4 = L.Convolution(ntrain.relu3, name='conv4', kernel_size=3, num_output=384, pad=1, group=2, weight_filler=dict(
        type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.1), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ntrain.relu4 = L.ReLU(ntrain.conv4, name='relu4', in_place=True)

    ntrain.conv5 = L.Convolution(ntrain.relu4, name='conv5', kernel_size=3, num_output=256, pad=1, group=2, weight_filler=dict(
        type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0.1), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ntrain.relu5 = L.ReLU(ntrain.conv5, name='relu5', in_place=True)
    ntrain.pool5 = L.Pooling(ntrain.relu5, name='pool5', kernel_size=3,
                             stride=2, pool=P.Pooling.MAX)

    ntrain.fc6 = L.InnerProduct(ntrain.pool5, name='fc6', num_output=4096,
                                weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=1e-1), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ntrain.relu6 = L.ReLU(ntrain.fc6, name='relu6', in_place=True)
    ntrain.drop6 = L.Dropout(ntrain.relu6, name='drop6',
                             dropout_ratio=0.5, in_place=True)

    ntrain.fc7 = L.InnerProduct(ntrain.drop6, name='fc7', num_output=4096,
                                weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=1e-1), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    ntrain.relu7 = L.ReLU(ntrain.fc7, name='relu7', in_place=True)
    ntrain.drop7 = L.Dropout(ntrain.relu7, name='drop7',
                             dropout_ratio=0.5, in_place=True)

    ntrain.fc8 = L.InnerProduct(ntrain.drop7, name='fc8', num_output=1000,
                                weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    ntrain.accuracy = L.Accuracy(ntrain.fc8, ntrain.label, name='accuracy',
                                 include=dict(phase=caffe.TEST))
    ntrain.loss = L.SoftmaxWithLoss(ntrain.fc8, ntrain.label, name='loss')

    #--------------------------------------------------
    # deploy，删去lr_mult、decay_mult、weight_filler、bias_filler
    # ({'shape': {'dim': [batch_size, channels, n_rows, n_cols]}})
    ndeploy.data = L.Input(input_param={'shape': {'dim': [10, 13, 227, 227]}})

    ndeploy.conv1 = L.Convolution(
        ndeploy.data, name='conv1', kernel_size=11, num_output=96, stride=4)
    ndeploy.relu1 = L.ReLU(ndeploy.conv1, name='relu1', in_place=True)
    ndeploy.norm1 = L.LRN(ndeploy.relu1, name='norm1',
                          local_size=5, alpha=1e-4, beta=0.75)
    ndeploy.pool1 = L.Pooling(ndeploy.norm1, name='pool1', kernel_size=3,
                              stride=2, pool=P.Pooling.MAX)

    ndeploy.conv2 = L.Convolution(
        ndeploy.pool1, name='conv2', kernel_size=5, num_output=256, pad=2, group=2)
    ndeploy.relu2 = L.ReLU(ndeploy.conv2, name='relu2', in_place=True)
    ndeploy.norm2 = L.LRN(ndeploy.relu2, name='norm2',
                          local_size=5, alpha=1e-4, beta=0.75)
    ndeploy.pool2 = L.Pooling(ndeploy.norm2, name='pool2', kernel_size=3,
                              stride=2, pool=P.Pooling.MAX)

    ndeploy.conv3 = L.Convolution(
        ndeploy.pool2, name='conv3', kernel_size=3, num_output=384, pad=1)
    ndeploy.relu3 = L.ReLU(ndeploy.conv3, name='relu3', in_place=True)

    ndeploy.conv4 = L.Convolution(
        ndeploy.relu3, name='conv4', kernel_size=3, num_output=384, pad=1, group=2)
    ndeploy.relu4 = L.ReLU(ndeploy.conv4, name='relu4', in_place=True)

    ndeploy.conv5 = L.Convolution(
        ndeploy.relu4, name='conv5', kernel_size=3, num_output=256, pad=1, group=2)
    ndeploy.relu5 = L.ReLU(ndeploy.conv5, name='relu5', in_place=True)
    ndeploy.pool5 = L.Pooling(ndeploy.relu5, name='pool5', kernel_size=3,
                              stride=2, pool=P.Pooling.MAX)

    ndeploy.fc6 = L.InnerProduct(ndeploy.pool5, name='fc6', num_output=4096)
    ndeploy.relu6 = L.ReLU(ndeploy.fc6, name='relu6', in_place=True)
    ndeploy.drop6 = L.Dropout(ndeploy.relu6, name='drop6',
                              dropout_ratio=5e-1, in_place=True)

    ndeploy.fc7 = L.InnerProduct(ndeploy.drop6, name='fc7', num_output=4096)
    ndeploy.relu7 = L.ReLU(ndeploy.fc7, name='relu7', in_place=True)
    ndeploy.drop7 = L.Dropout(ndeploy.relu7, name='drop7',
                              dropout_ratio=5e-1, in_place=True)

    ndeploy.fc8 = L.InnerProduct(ndeploy.drop7, name='fc8', num_output=1000)

    ndeploy.prob = L.Softmax(ndeploy.fc8, name='prob')

    out_train_val = str('name: "AlexNet"\n') + \
        str(nval.to_proto()) + str(ntrain.to_proto())
    with open(output_path + '/alexnet_train_val.prototxt', 'w') as f:
        f.write(out_train_val)

    out_deploy = str('name: "AlexNet"\n') + str(ndeploy.to_proto())
    with open(output_path + '/alexnet_deploy.prototxt', 'w') as f:
        f.write(out_deploy)


alexnet('examples/imagenet/ilsvrc12_train_lmdb', 'examples/imagenet/ilsvrc12_val_lmdb', 'data/ilsvrc12/imagenet_meantrain.binaryproto',
        256, 50, '/home/zxh')
