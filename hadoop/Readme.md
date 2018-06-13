# Hadoop的一些经验技巧

## Hadoop的安装
在Apache官方网站下载最新的Hadoop，解压，有两种运行模式：   
1)单机模式
```shell
/home/zxh/hadoop-3.0.3/bin/hadoop jar /home/zxh/hadoop-3.0.3/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.0.3.jar grep /home/zxh/input /home/zxh/output 'dfs[a-z.]+'
```
