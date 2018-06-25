# Hadoop的一些经验技巧

## 伪分布式Hadoop的安装、配置
在Apache官方网站下载最新的Hadoop，解压，有两种运行模式：   
(1)检查/etc/hosts，必须形如以下的格式
```text
192.168.1.1       ubuntu
127.0.1.1         localhost
```
(2)创建hadoop用户
```shell
useradd -d /home/hadoop -s /bin/bash -m hadoop
passwd hadoop
```
(3)为hadoop用户设置ssh免登录
```shell
ssh-keygen -t rsa
cat /home/hadoop/.ssh/id_rsa.pub >> /home/hadoop/.ssh/authorized_keys
```
(4)将hadoop软件包上传到hadoop用户下，解压，在解压后的目录中创建新目录
```shell
mkdir /home/hadoop/hadoop-3.0.3/tmp
mkdir -p /home/hadoop/hadoop-3.0.3/hdfs/name
mkdir -p /home/hadoop/hadoop-3.0.3/hdfs/data
```
(5)修改 /home/hadoop/hadoop-3.0.3/etc/hadoop/core-site.xml
```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>file:/home/hadoop/hadoop-3.0.3/tmp</value>
    </property>
</configuration>
```
(6)修改 /home/hadoop/hadoop-3.0.3/etc/hadoop/hdfs-site.xml
```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:/home/hadoop/hadoop-3.0.3/hdfs/name</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:/home/hadoop/hadoop-3.0.3/hdfs/data</value>
    </property>
    <property>
        <name>dfs.http.address</name>
        <value>192.168.1.1:50070</value>
    </property>
</configuration>
```
(7)修改 /home/hadoop/hadoop-3.0.3/etc/hadoop/mapred-site.xml
```xml
<configuration>
    <property>
        <!--告诉Hadoop，MapReduce运行在Yarn上-->
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
```
(8)修改 /home/hadoop/hadoop-3.0.3/etc/hadoop/yarn-site.xml
```xml
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
	<property>
        <name>yarn.resourcemanager.hostname</name>
        <value>localhost</value>
    </property>
	<property>
        <name>yarn.log-aggregation-enable</name>
        <value>true</value>
    </property>
</configuration>
```
(9)设置环境变量
```shell
export HADOOP_HOME=/home/hadoop/hadoop-3.0.3
export PATH=${HADOOP_HOME}/bin:${PATH}
```
(10)初始格式化HDFS
```shell
hdfs namenode -format
```
(11)启动HDFS
```shell
start-dfs.sh
```
