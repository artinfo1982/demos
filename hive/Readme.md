# hive的一些使用技巧
## hive直接导入本地txt文件完成快速数据预置
1.创建hive表
```text
create table a (id int, a1 string, a2 string) row format delimited fields terminated by ',' lines terminated by '\n' stored as textfile;
```
2.创建txt文件，取名a.txt，可以手工写入，也可以程序写入
```text
1,a1,a2
2,b1,b2
3,c1,c2
```
3.将a.txt上传到hive对应的表路径下
```text
hadoop fs -put /home/a.txt /user/hive/warehouse/test.db/a/
```
