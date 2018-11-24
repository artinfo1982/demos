#coding=utf-8

from caffe.proto import caffe_pb2

def toPrototxt(modelName, deployName):
  with open(modelName, 'rb') as f:
    caffemodel = caffe_pb2.NetParameter()
    caffemodel.ParseFromString(f.read())
    
  for item in caffemodel.layers:
    item.ClearField('blobs')
  for item in caffemodel.layer:
    item.ClearField('blobs')
    
  with open(deployName, 'w') as f:
    f.write(str(caffemodel))
    
if __name__ == '__main__':
  '''
  VGG 16 caffemodel下载地址：
  http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
  '''
  modelName = 'VGG_ILSVRC_16_layers.caffemodel'
  deployName = 'VGG_ILSVRC_16_layers.prototxt'
  toPrototxt(modelName, deployName)
