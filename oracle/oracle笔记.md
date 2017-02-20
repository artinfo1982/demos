清理用户和表空间的步骤：   
1. 删除用户的表空间   
drop tablespace XXX including contents and datafiles;   
2. 删除用户及其关联项：   
drop user XXX cascade;   
   
检查用户表空间是否存在：   
select * from v$dbfie order by 1;   
   
关闭oracle的log（例如不写undo）：   
alter table XXX nologging;   
   
创建大于等于32G表空间时的bigfile选项：   
create bigfile tablespace XXX datafile '/home/oracle/abc.dbf' size 1G autoextend on maxsize 32G;
