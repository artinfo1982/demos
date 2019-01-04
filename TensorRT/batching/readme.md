根据若干图片，生成batches文件。   
使用方法：   
```shell
g++ generate_batches.cpp -o generate_batches -std=c++11 -O2 `pkg-config --libs opencv`
./generate_batches images/ batches/ result.txt
```
images文件夹中是若干原始图片   
batches文件夹中存放生成的batch文件   
result.txt是labels列表，一行对应一个图片的label

batch文件结构说明：   
```text
每个batch文件由3个二进制块block1 + block2 + block3组成。
block1：N, C, H, W，4个整数，分别表示batch_size，channel，high，width。
block2：如果channel=3，就是N个图片的r、g、b数组；如果channel=1，那就是N个灰度值数组，数据类型都是float。
block3：N个图片的N个label，每个图片对应一个label，label的数据类型是float。
```
