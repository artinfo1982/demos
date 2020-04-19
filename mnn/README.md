# mnn的编译和使用方法

## 依赖
1. cmake
```text
cmake安装最新版
```
2. protobuf
```shell
sudo apt install autoconf automake libtool curl make g++ unzip
git clone https://github.com/google/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
```

## 下载mnn
下载指定版本的mnn
```text
https://github.com/alibaba/MNN/releases
```
或者下载github上的最新代码
```shell
git clone https://github.com/alibaba/MNN.git
```

## 编译mnn x86_64的模型转换工具、量化工具
```shell
tar -zxvf MNN-xx.xx.xx.xx.tar.gz
cd MNN-xx.xx.xx.xx
cd schema
./generate.sh
mkdir build
cd build
cmake .. -DMNN_BUILD_TOOLS=false -DMNN_BUILD_QUANTOOLS=true -DMNN_BUILD_CONVERTER=true
# 注：其余的cmake参数，参见https://www.yuque.com/mnn/cn/dvvocw
make -j 8
# 编译成功后，在build目录会生成MNNConvert, quantized.out等可执行程序
```

## 编译mnn arm32/64的benchmark工具(假设不使用GPU)
vi MNN-xx.xx.xx.xx/benchmark/bench_android.sh
```text
OPENMP="ON" --> OPENMP="OFF"
VULKAN="ON" --> VULKAN="OFF"
OPENCL="ON" --> OPENCL="OFF"
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake --> -DCMAKE_TOOLCHAIN_FILE=/home/cd/android-ndk-r16b/build/cmake/android.toolchain.cmake
./bench_android.sh -64 # 如果编译32位，则不加-64
```
