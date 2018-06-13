# Hadoop的一些经验技巧

## Hadoop的安装
在Apache官方网站下载最新的Hadoop，解压，有两种运行模式：   
1)单机模式
```shell
#创建输入文件文件夹，用于存放需要处理的数据文件
mkdir /home/zxh/input
#将etc/hadoop下面的xml文件放入输入文件夹，模拟作为处理的源
cp /home/zxh/hadoop-3.0.3/etc/hadoop/*.xml /home/zxh/input
#执行数据处理，此处使用grep模拟
/home/zxh/hadoop-3.0.3/bin/hadoop jar /home/zxh/hadoop-3.0.3/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.0.3.jar grep /home/zxh/input /home/zxh/output 'dfs[a-z.]+'
```
