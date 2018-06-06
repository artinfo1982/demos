# TVM & NNVM用法总结

## 编译安装NNVM、TVM
如果想让TVM、NNVM支持x86架构的CPU和ARM，就必须先安装LLVM。   
1.添加LLVM安装源
```shell
#以ubuntu系统为例，在 /etc/apt/sources.list 的最后，添加最新的LLVM源（以6.0版本为例）
#安装源的地址，可以参考 https://apt.llvm.org/
deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main
deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main
```
2.安装LLVM
```shell
sudo apt install clang-6.0 lldb-6.0
```
3.下载TVM、NNVM
```shell
git clone --recursive https://github.com/dmlc/nnvm
```
4.编译安装TVM
```shell
cd ~/nnvm/tvm
cp make/config.mk .
#修改config.mk，将其中的"LLVM_CONFIG = "改为"LLVM_CONFIG = llvm-config-6.0"
make
```
5.安装TVM和TOPI的python包
```shell
#在.bashrc中设置PYTHONPATH
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm/topi/python:${PYTHONPATH}
source .bashrc
cd ~/nnvm/tvm/python
python setup.py install
cd ~/nnvm/tvm/topi/python
python setup.py install
```
6.编译安装NNVM
```shell
cd ~/nnvm
cp make/config.mk .
make
```
7.安装NNVM的python包
```shell
#在.bashrc中将NNVM的Python路径添加到PYTHONPATH
export PYTHONPATH=/path/to/tvm/python:/path/to/tvm/topi/python:/path/to/nnvm/python:${PYTHONPATH}
source .bashrc
cd ~/nnvm/python
python setup.py install
```
