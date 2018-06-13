# Hadoop的一些经验技巧

## Hadoop的安装
在Apache官方网站下载最新的Hadoop，解压，有两种运行模式：   
1)单机
```shell
#创建输入文件文件夹，用于存放需要处理的数据文件
mkdir ~/input
#将etc/hadoop下面的xml文件放入输入文件夹，模拟作为处理的源
cp ~/hadoop-3.0.3/etc/hadoop/*.xml ~/input
#执行数据处理，此处使用grep模拟
~/hadoop-3.0.3/bin/hadoop jar ~/hadoop-3.0.3/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.0.3.jar grep ~/input ~/output 'dfs[a-z.]+'
```
2)伪分布式（最常使用）
假设 master: 192.168.1.1，slave1: 192.168.1.2，slave2: 192.168.1.3   
在master的机器上修改 ~/hadoop-3.0.3/etc/hadoop/core-site.xml
```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://192.168.1.1:9000</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>~/hadoop-3.0.3/tmp</value>
    </property>
</configuration>
```
