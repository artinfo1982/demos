针对caffemodel和prototxt的faster rcnn的fp32、int8实现。   

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
g++ fasterRCNN_fp32.cpp common.h common.cpp -I"/home/cd/TensorRT-4.0.1.6/include" -I"/usr/local/cuda/include" -I"/usr/local/include" -Wall -std=c++11 -L"../../lib" -L"/usr/local/cuda/lib64" -L"/usr/local/lib" -L"../lib" -lnvinfer -lnvparsers -lnvinfer_plugin -lcublas -lcudart -lrt -ldl -lpthread `pkg-config --libs opencv` -o faster_rcnn_fp32 -O3
g++ fasterRCNN_int8.cpp common.h common.cpp data_loader.h -I"/home/cd/TensorRT-4.0.1.6/include" -I"/usr/local/cuda/include" -I"/usr/local/include" -Wall -std=c++11 -L"../../lib" -L"/usr/local/cuda/lib64" -L"/usr/local/lib" -L"../lib" -lnvinfer -lnvparsers -lnvinfer_plugin -lcublas -lcudart -lrt -ldl -lpthread `pkg-config --libs opencv` -o faster_rcnn_int8 -O3
```

运行：
```shell
./faster_rcnn_fp32
./faster_rcnn_int8
```

结果：
```text
在NVIDIA Tesla P4 GPU上的测试结果：

fp32:
avg infer time of each image=138.291ms, top1 error rate=24%,top5 error rate=15.0667%, total number=3000, total top1_success=2280, total top5_success=2548

int8:
avg infer time of each image=55.7156ms, top1 error rate=24.2%,top5 error rate=15.5667%, total number=3000, total top1_success=2274, total top5_success=2533

压缩前的fp32权值文件大小（byte）：548317115
压缩后的int8权值文件大小（byte）：138687816
压缩后的文件只有原始文件的1/4
```

注意点：
```text
1. fasterRCNN_int8.cpp中用到的/home/cd/TensorRT-4.0.1.6/data/faster-rcnn/list.txt的内容为PASCAL VOC图片集每一张图片的绝对路径，一行对应一个文件。
2. 上述代码，仅在TensorRT-4.0.1.6 + cuda 9.2 + cudnn 7.1下测试通过。
3. 在NVIDIA Tesla P4 GPU上的测试数据。
```
