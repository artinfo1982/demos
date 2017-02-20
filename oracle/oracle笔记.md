清理用户和表空间的步骤：   
1. 删除用户的表空间   
   drop tablespace XXX including contents and datafiles;   
2. 删除用户及其关联项：   
   drop user XXX cascade;
