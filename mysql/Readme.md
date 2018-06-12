# mysql的若干经验技巧

## 创建用户
```sql
create user 'cd'@'%' identified by 'huawei';
```

## 创建数据库
```sql
create database if not exists cddb DEFAULT CHARSET utf8 COLLATE utf8_general_ci;
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

## 创建表
```sql
use cddb;
create table if not exists cd_tb_test
(
	v_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
	-- 单精度浮点，共5位有效数字，其中2位小数，最后一位四舍五入
	v_float FLOAT(5, 2),
	-- 双精度浮点，共5位有效数字，其中2位小数，最后一位四舍五入
	v_double DOUBLE(5, 2),
	v_char CHAR(1),
	v_varchar VARCHAR(64) NOT NULL,
	-- 日期，yyyy-MM-dd
	v_date DATE,
	-- 日期+时间，yyyy-MM-dd HH:mm:ss
	v_datetime DATETIME,
	v_blob BLOB,
	PRIMARY KEY ( v_id )
)
ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

## 查看当前数据库中有哪些表
```text
mysql> show tables;
+----------------+
| Tables_in_cddb |
+----------------+
| cd_tb_test     |
+----------------+
1 row in set (0.00 sec)
```

## 查看某一张表的详情
```text
mysql> desc cd_tb_test;
+------------+------------------+------+-----+---------+----------------+
| Field      | Type             | Null | Key | Default | Extra          |
+------------+------------------+------+-----+---------+----------------+
| v_id       | int(10) unsigned | NO   | PRI | NULL    | auto_increment |
| v_float    | float(5,2)       | YES  |     | NULL    |                |
| v_double   | double(5,2)      | YES  |     | NULL    |                |
| v_char     | char(1)          | YES  |     | NULL    |                |
| v_varchar  | varchar(64)      | NO   |     | NULL    |                |
| v_date     | date             | YES  |     | NULL    |                |
| v_datetime | datetime         | YES  |     | NULL    |                |
| v_blob     | blob             | YES  |     | NULL    |                |
+------------+------------------+------+-----+---------+----------------+
8 rows in set (0.01 sec)
```

## mysql存储过程示例
```sql
CREATE DEFINER=`cd`@`%` PROCEDURE `cd_proc_test`()
BEGIN
    DECLARE i INT UNSIGNED;
    DECLARE v_float FLOAT(5, 2);
    DECLARE v_double DOUBLE(5, 2);
    DECLARE v_char CHAR(1);
    DECLARE v_varchar VARCHAR(64);
    DECLARE v_date DATE;
    DECLARE v_datetime DATETIME;
    DECLARE v_blob BLOB;
    
    -- 大小写+数字的字母集合，供后续的随机字符串使用
    DECLARE v_Aa0 char(62) default 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    
    SET i = 1;
    WHILE i <= 10 DO
    	-- 100以内的随机数，包含小数
        SELECT RAND()*100 into v_float;
	-- 100以内的随机数，包含小数
	SELECT RAND()*100 into v_double;
	-- 随机生成一位字符
	SET v_char = SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1);
	-- 随机生成6位字符串（包含大小写），其实就是6个单一字符的生成函数的拼装
	SET v_varchar = CONCAT(SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1));
	-- 获取当前日期，yyyy-MM-dd
	SET v_date = CURDATE();
	-- 获取当前日期+时间，yyyy-MM-dd HH:mm:ss
	SET v_datetime = SYSDATE();
	-- blob类型变量的使用，具体内容可以替换为图片，和oracle类似
	SET v_blob = HEX('H6rxHQ1dgA');
	INSERT INTO cd_tb_test values(i, v_float, v_double, v_char, v_varchar, v_date, v_datetime, v_blob);
	SET i = i + 1;
    END WHILE;
END;
```
使用mysql-front，在数据库上右键“新建”-->“过程”，将存储过程copy进去。   
使用存储过程的时候，右键点击存储过程，选择“打开一个新的窗口”，点击绿色“运行”按钮。
