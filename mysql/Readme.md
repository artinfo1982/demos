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
	v_vchar1 VARCHAR(64) NOT NULL,
	v_vchar2 VARCHAR(64) NOT NULL,
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
| v_vchar1   | varchar(64)      | NO   |     | NULL    |                |
| v_vchar2   | varchar(64)      | NO   |     | NULL    |                |
| v_date     | date             | YES  |     | NULL    |                |
| v_datetime | datetime         | YES  |     | NULL    |                |
| v_blob     | blob             | YES  |     | NULL    |                |
+------------+------------------+------+-----+---------+----------------+
9 rows in set (0.01 sec)
```

## mysql存储过程示例
```sql
CREATE DEFINER=`cd`@`%` PROCEDURE `cd_proc_test`()
BEGIN
    DECLARE i INT UNSIGNED;
    DECLARE v_float FLOAT(5, 2);
    DECLARE v_double DOUBLE(5, 2);
    DECLARE v_char CHAR(1);
    DECLARE v_vchar1 VARCHAR(64);
    DECLARE v_vchar2 VARCHAR(64);
    DECLARE v_date DATE;
    DECLARE v_datetime DATETIME;
    DECLARE v_blob BLOB;
    
    -- 大小写+数字的字母集合，供后续的随机字符串使用
    DECLARE v_Aa0 CHAR(62) DEFAULT 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    -- 定义英文字符串数组
    DECLARE v_array_eng VARCHAR(16) DEFAULT 'aaa bbb ccc';
    -- 定义中文单词数组
    DECLARE v_array_chn VARCHAR(64) DEFAULT '中国 韩国 日本';
    
    SET i = 1;
    WHILE i <= 10 DO
    	-- 100以内的随机数，包含小数
        SELECT RAND()*100 INTO v_float;
	-- 100以内的随机数，包含小数
	SELECT RAND()*100 INTO v_double;
	-- 随机生成一位字符
	SET v_char = SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1);
	-- 随机生成6位字符串（包含大小写），其实就是6个单一字符的生成函数的拼装
	SET v_vchar1 = CONCAT(SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1),SUBSTRING(v_Aa0,1+FLOOR(RAND()*61),1));
	-- 获取当前日期，yyyy-MM-dd
	SET v_date = CURDATE();
	-- 获取当前日期+时间，yyyy-MM-dd HH:mm:ss
	SET v_datetime = SYSDATE();
	-- blob类型变量的使用，具体内容可以替换为图片，和oracle类似
	SET v_blob = HEX('H6rxHQ1dgA');
	IF MOD(i, 2) = 0 THEN
	    -- 轮询抽取英文数组中的元素，以空格为分隔符
	    SET v_vchar2 = SUBSTRING_INDEX(SUBSTRING_INDEX(v_array_eng, ' ', MOD(i, 3) + 1),' ',-1);
	ELSE
	    -- 轮询抽取中文数组中的元素，以空格为分隔符
	    SET v_vchar2 = SUBSTRING_INDEX(SUBSTRING_INDEX(v_array_chn, ' ', MOD(i, 3) + 1),' ',-1);
	END IF;
	INSERT INTO cd_tb_test values(i, v_float, v_double, v_char, v_vchar1, v_vchar2, v_date, v_datetime, v_blob);
	SET i = i + 1;
	IF MOD(i, 5) = 0 THEN
	    COMMIT;
	END IF;
    END WHILE;
    COMMIT;
END;
```
使用mysql-front，在数据库上右键“新建”-->“过程”，将存储过程copy进去。   
使用存储过程的时候，右键点击存储过程，选择“打开一个新的窗口”或者“打开一个新的标签页”，点击绿色“运行”按钮。   
执行上述存储过程后，查询表：
```text
mysql> select * from cd_tb_test;
+------+---------+----------+--------+----------+----------+------------+---------------------+----------------------+
| v_id | v_float | v_double | v_char | v_vchar1 | v_vchar2 | v_date     | v_datetime          | v_blob               |
+------+---------+----------+--------+----------+----------+------------+---------------------+----------------------+
|    1 |   81.71 |    68.94 | 8      | 3IfTEn   | 韩国     | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    2 |   60.07 |    35.27 | 6      | T0dSBf   | ccc      | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    3 |    8.07 |    13.57 | A      | VIEYMM   | 中国     | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    4 |   29.73 |    58.62 | c      | Ad8XcX   | bbb      | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    5 |   92.51 |    18.44 | i      | kCUzVQ   | 日本     | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    6 |    8.94 |    38.04 | M      | boh4nu   | aaa      | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    7 |    5.25 |    23.01 | 8      | qyjJDL   | 韩国     | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    8 |   66.72 |    47.67 | x      | DpZAM1   | ccc      | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|    9 |   47.72 |    75.93 | w      | HMHYGi   | 中国     | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
|   10 |    9.11 |     4.97 | 7      | SRxUPh   | bbb      | 2018-06-12 | 2018-06-12 22:13:08 | 48367278485131646741 |
+------+---------+----------+--------+----------+----------+------------+---------------------+----------------------+
10 rows in set (0.00 sec)
```

## mysql快速插入亿级数据
```sql
--***非常重要***
--注意，在插入前，一定要先执行 alter table XXX ENGINE = MYISAM; 加快插入速度
--在插入都完成后，一定要执行 alter table XXX ENGINE = INNODB; 改回来
CREATE DEFINER=`cd`@`%` PROCEDURE `cd_proc_test`()
BEGIN
    DECLARE i INT UNSIGNED;
    DECLARE v_vchar VARCHAR(64);
    
    SET v_vchar = 'aaa';
    SET i = 1;
    
    WHILE i <= 100000000 DO
	INSERT INTO cd_tb_test values(i, v_vchar);
	SET i = i + 1;
	IF MOD(i, 10000) = 0 THEN
	    COMMIT;
	END IF;
    END WHILE;
    COMMIT;
END;
```

## mysql二进制高速迁移方案
```text
假设我们需要将mysql A-->B
1.在B上创建和A同名的database
2.在B上建表
3.停掉B的mysql
4.在B的mysql配置文件my.cnf中的mysqld配置段中添加以下两行：
innodb_file_per_table = 1
innodb_force_recovery = 1
5.将B中mysql数据目录下数据库名目录下的.ibd文件都删了
6.将A对应目录下的.ibd文件复制到一个临时目录中
7.分别将临时目录中所有ibd文件的第37-38字节、第41-42字节的内容替换为A对应的ibd文件的第37-38字节内容，详见下面的C代码：
https://github.com/artinfo1982/demos/blob/master/mysql/migrate/mod_ibd.c
8.将临时目录中所有的ibd文件都复制到B相应的目录中
9.启动B
```
