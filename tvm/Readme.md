# TVM & NNVM用法总结

## 编译安装NNVM、TVM
如果想让TVM、NNVM支持x86架构的CPU和ARM，就必须先安装LLVM。
1.编译安装LLVM
```shell
#以ubuntu系统为例，在 /etc/apt/sources.list 的最后，添加最新的LLVM源（以6.0版本为例），可以参考 https://apt.llvm.org/
deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main
deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-6.0 main
```
