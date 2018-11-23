1.执行如下命令生成test.pb.h、test.pb.cc
```shell
protoc test.proto --cpp_out .
```

2.编写Main.cpp，序列化、反序列化示例

3.make之后，运行，生成pb文件，并打印出解析pb文件内容。
```text
zxh@ubuntu:~/protobuf$ hexdump -C addressBook.pb
00000000  0a 21 0a 01 61 10 01 1a  09 61 40 31 32 36 2e 63  |.!..a....a@126.c|
00000010  6f 6d 22 0f 0a 0b 31 33  39 31 33 39 30 34 30 36  |om"...1391390406|
00000020  32 10 00 0a 22 0a 01 62  10 02 1a 09 62 40 31 32  |2..."..b....b@12|
00000030  36 2e 63 6f 6d 22 10 0a  0c 30 32 35 2d 35 36 36  |6.com"...025-566|
00000040  32 30 30 30 30 10 02                              |20000..|
00000047
```
```text
zxh@ubuntu:~/protobuf$ ./test
----------------------------------
Person ID: 1
Person Name: a
E-mail address: a@126.com
Mobile phone: 13913904062
----------------------------------

----------------------------------
Person ID: 2
Person Name: b
E-mail address: b@126.com
Work phone: 025-56620000
----------------------------------
```
