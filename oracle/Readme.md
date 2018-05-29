# oracle的一些使用技巧
## 创建表空间时使用bigfile
```sql
create bigfile tablespace XXX datafile '/home/oracle/abc.dbf' size 1G autoextend on maxsize 32G;
```
