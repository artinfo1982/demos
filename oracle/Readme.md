# oracle的一些使用技巧
## 创建表空间时使用bigfile
```sql
create bigfile tablespace XXX datafile '/home/oracle/abc.dbf' size 1G autoextend on maxsize 32G;
```
## 针对某张表启用nologging
```sql
alter table XXX nologging;
```
## 导入导出
```sql
--导出某些表
exp 用户名/密码@ip:port/SID file=/xx/test.dmp tables=XX,YY compress=y
--导出整个用户下的所有东西
exp 用户名/密码@ip:port/SID file=/xx/test.dmp full=y compress=y
--导入
imp 用户名/密码@ip:port/SID file=/xx/test.dmp tables=XX,YY
```
## 快速将文本导入表
```sql
--方法1
insert into XXX (id,name)
select 1,'a' from dual union all
select 2,'b' from dual union all
select 3,'c' from dual;

--方法2
--使用oracle自带的工具sqlldr
--sqlldr用法示例：
--1. 假设存在一张如下的表
create table user (
  id number,
  name varchar2(20)
)
--2. 创建一个文本文件test.txt，用于存放需要插入上表的数据
  cat test.txt
  1,张三
  2,李四
  3,王五
--3. 创建一个控制文件test.ctl，用于控制sqlldr的行为
  cat test.ctl
  load data
  infile 'test.txt'
  insert into table user
  fields terminated by ","
  (id,name)
--4. 在oracle用户下执行
sqlldr userid=userName/passwd@oracleIP:oraclePort/oracleSID control=test.ctl silent=header,feedback
--如果有多个文件，可以多进程后台执行sqlldr
```
## 检查用户表空间是否存在
```sql
select * from v$dbfie order by 1;
```
## 清理用户和表空间
```sql
--1. 删除用户的表空间
drop tablespace XXX including contents and datafiles;
--2. 删除用户及其关联项：
drop user XXX cascade;
```
## oracle存储过程
```sql
CREATE OR REPLACE PROCEDURE XXX
AS
  --定义数组，如果是中文字符，注意每个占3个字节
  type array is table of VARCHAR2(400);
  test_array := array('人','口','手');
  --变量定义
  id VARCHAR2(8);
  ...
BEGIN
  a := '1';
  ...
FOR i IN 1..10
LOOP
  --随i自增，共8位，左补0
  id := lpad(i, 8, 0);
  IF MOD(i, 2) = 0 THEN
    ...
  ELSE
    ...
  END IF;
  
  --2位大写字母+4个数字，U表示大写，X表示大写或者数字，随机整数：trunc(dbms_random.value(0, XXX))
  b := dbms_random.string('U', 2) || lpad(trunc(dbms_random.value(0, 9999)), 4, 0);
  --取数组元素
  c := test_array(2);
  
  --if-else if
  IF XX THEN
    ...
  ELSE IF XX THEN
    ...
  ELSE IF XX THEN
    ...
  END IF;
  END IF;
  END IF;
  
  --对于blob类型，假设d是blob类型
  d := HEXTORAW('十六进制字符串');
  
  INSERT INTO XX values (id, xx, xx, ...);
  
  IF MOD(i, 10000) = 0 THEN
    COMMIT;
  END IF;
END LOOP;
COMMIT;

END;
```
## oracle快速插入亿级数据
```sql
CREATE OR REPLACE PROCEDURE XXX AS
BEGIN
  FOR i IN 1..100000000 LOOP
    --启用append，不写日志，并行插入，并行处理的线程数为10
    INSERT INTO /*+ append parallel(T_TEST,10) nologging */ T_TEST values(i);
    IF MOD(i,1000000) = 0 THEN
      COMMIT;
    END IF;
  END LOOP;
END;
```
