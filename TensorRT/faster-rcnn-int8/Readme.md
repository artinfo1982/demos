采用int8，仿照官网的sampleINT8、sampleFasterRCNN，改写适用int8的faster rcnn。   

faster rcnn权值文件caffemodel下载：
```shell
wget --no-check-certificate https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0 -O data/faster-rcnn/faster-rcnn.tgz
```

解压：
```shell
tar zxvf data/faster-rcnn/faster-rcnn.tgz -C data/faster-rcnn --strip-components=1 --exclude=ZF_*
```

编译：
```shell
g++ fasterRCNN_int8.cpp common.h common.cpp BatchStream.h data_loader.h -I"/home/cd/TensorRT/include" -I"/usr/local/cuda/include" -I"/usr/local/include" -Wall -std=c++11 -L"../../lib" -L"/usr/local/cuda/targets/x86_64-linux/lib64" -L"/usr/local/lib" -L"../lib" -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread `pkg-config --libs opencv` -o sample_faster_rcnn_int8
```
