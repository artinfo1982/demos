# hive的一些使用技巧
## txt文件数据导入hive
假设 /home/aaa/data.txt 中的内容为：
```text
1,a
2,b
3,c
4,d
...
```
在hive上建表
```shell
hive>create table t_test(id int, name string) row format delimited fields terminated by ',';
```
导入txt数据
```shell
hive>load data local inpath '/home/aaa/data.txt' overwrite into t_test;
```
