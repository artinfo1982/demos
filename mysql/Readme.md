# mysql的若干经验技巧

## 创建用户
```sql
create user 'cd'@'%' identified by 'huawei';
```

## 创建数据库
```sql
create database cddb DEFAULT CHARSET utf8 COLLATE utf8_general_ci;
```

## 给用户操作数据库的权限
```sql
grant all privileges on `cddb`.* to 'cd'@'%' identified by 'huawei';
```

## 刷新配置
```sql
flush privileges;
```

## 查看当前所有数据库
```text
mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| cddb               |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
5 rows in set (0.00 sec)
```

## 查看当前所有用户
```text
mysql> SELECT DISTINCT CONCAT('User: ''',user,'''@''',host,''';') AS query FROM mysql.user;
+---------------------------------------+
| query                                 |
+---------------------------------------+
| User: 'cd'@'%';                       |
| User: 'debian-sys-maint'@'localhost'; |
| User: 'mysql.session'@'localhost';    |
| User: 'mysql.sys'@'localhost';        |
| User: 'root'@'localhost';             |
+---------------------------------------+
5 rows in set (0.00 sec)
```

## 查看某个具体用户的权限
```text
mysql> show grants for 'cd'@'%';
+----------------------------------------------+
| Grants for cd@%                              |
+----------------------------------------------+
| GRANT USAGE ON *.* TO 'cd'@'%'               |
| GRANT ALL PRIVILEGES ON `cddb`.* TO 'cd'@'%' |
+----------------------------------------------+
2 rows in set (0.00 sec)
```
