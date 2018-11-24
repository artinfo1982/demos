#coding=utf-8

from caffe.proto import caffe_pb2

def toPrototxt(modelName, deployName):
  with open(modelName, 'rb') as f:
    caffemodel = caffe_pb2.NetParameter()
