采用int8，仿照官网的sampleINT8、sampleFasterRCNN，改写适用int8的faster rcnn。

faster rcnn权值文件caffemodel下载：
wget --no-check-certificate https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0 -O data/faster-rcnn/faster-rcnn.tgz

解压：
tar zxvf data/faster-rcnn/faster-rcnn.tgz -C data/faster-rcnn --strip-components=1 --exclude=ZF_*
