# Hadoop的一些经验技巧

## 伪分布式Hadoop的安装、配置
在Apache官方网站下载最新的Hadoop，解压，有两种运行模式：   
**(1)检查/etc/hosts，必须形如以下的格式
```text
192.168.1.1       ubuntu
127.0.1.1         localhost
```
**(2)
伪分布式（最常使用）
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
