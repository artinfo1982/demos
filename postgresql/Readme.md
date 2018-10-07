# postgresql的若干经验技巧

## postgresql从本地txt导入数据到表
1.创建表
```text
create table a (id int, a1 varchar2(50));
```
2.准备本地txt文件，例如/home/a.txt，写入数据，可以手工写入，也可以程序写入，示例：
```text
1,a
2,b
```
3.执行如下命令
```text
copy a from '/home/a.txt' with (format text, delimiter ',');
```
