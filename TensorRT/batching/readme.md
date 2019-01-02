根据若干图片，生成batches文件。   
使用方法：   
```shell
g++ generate_batches.cpp -o generate_batches -std=c++11 -O2 `pkg-config --libs opencv`
./generate_batches images/ batches/ result.txt
```
images文件夹中是若干原始图片   
batches文件夹中存放生成的batch文件   
result.txt是labels列表，一行对应一个图片的label
