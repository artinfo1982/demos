# mnn的编译和使用方法

## 下载mnn
下载指定版本的mnn
```text
https://github.com/alibaba/MNN/releases
```
或者下载github上的最新代码
```shell
git clone https://github.com/alibaba/MNN.git
```

## 编译mnn x86_64工具
```shell
tar -zxvf MNN-xx.xx.xx.xx.tar.gz
cd MNN-xx.xx.xx.xx
cd schema
./generate.sh
mkdir build
cd build
cmake .. -DMNN_BUILD_TOOLS=false -DMNN_BUILD_QUANTOOLS=true -DMNN_BUILD_CONVERTER=true
```
