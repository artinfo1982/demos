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
