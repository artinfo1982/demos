1.执行如下命令生成test.pb.h、test.pb.cc
```shell
protoc test.proto --cpp_out .
```

2.编写Main.cpp，序列化、反序列化示例

3.make之后，运行，生成pb文件，并打印出解析pb文件内容。
