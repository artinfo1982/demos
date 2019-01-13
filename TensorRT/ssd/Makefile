ssd_int8:
  nvcc -o libsoftmax.so -shared -Xcompiler -fPIC softmax.cu
  g++ ssd_int8.cpp ssd_iplugin.h ssd_iplugin.cpp common.h common.cpp data_loader.h -I"/home/cd/TensorRT-4.0.1.6/include" -I"/usr/local/cuda/include" -I"/usr/local/include" -Wall -std=c++11 -L"../../lib" -L"/usr/local/cuda/lib64" -L"/usr/local/lib" -L"../lib" -L. -lsoftmax -lnvinfer -lnvparsers -lnvinfer_plugin -lcublas -lcudart -lrt -ldl -lpthread `pkg-config --libs opencv` -o ssd_int8
  
clean:
  rm ssd_int8 libsoftmax.so
