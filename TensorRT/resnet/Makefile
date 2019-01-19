resnet50:
  g++ resnet50_int8.cpp common.h common.cpp data_loader.h -I"/home/cd/TensorRT-4.0.1.6/include" -I"/usr/local/cuda/include" -I"/usr/local/include" -Wall -std=c++11 -L"../../lib" -L"/usr/local/cuda/lib64" -L"/usr/local/lib" -L"../lib" -lnvinfer -lnvparsers -lnvinfer_plugin -lcublas -lcudart -lrt -ldl -lpthread `pkg-config --libs opencv` -o resnet50_int8 -O3
  g++ resnet50_fp32.cpp common.h common.cpp -I"/home/cd/TensorRT-4.0.1.6/include" -I"/usr/local/cuda/include" -I"/usr/local/include" -Wall -std=c++11 -L"../../lib" -L"/usr/local/cuda/lib64" -L"/usr/local/lib" -L"../lib" -lnvinfer -lnvparsers -lnvinfer_plugin -lcublas -lcudart -lrt -ldl -lpthread `pkg-config --libs opencv` -o resnet50_fp32 -O3
  
clean:
  rm resnet50_int8 resnet50_fp32
