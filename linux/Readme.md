# Linux若干技巧
## root无法登录重置root密码
```text
1. 进入救援模式：启动options填写为 init=/bin/bash
2. mount -n / -o remount,rw
3. /usr/bin/passwd 给root重置密码
4. mount -n / -o remount,ro
5. exit
6. 重启
```
## linux挂载远程windows服务器的路径
```shell
mount -t cifs -o username=xx,password=xx //192.168.1.1/test /opt/aaa
```
## sendmail
```text
1. 配置mail.rc
set asksub append dot save crt=20
ignore Received Message-Id Resent-Message-Id Status Mail-From Return-Path Via
set from = 发件人邮箱地址
set smtp = mail.huawei.com
set smtp-auth-user = XXXX
set smtp-auth-password = XXXX
set smtp-auth = login

2.echo "邮件正文" | mail -s "邮件标题" aa@126.com,bb@gmail.com
```
