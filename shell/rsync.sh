


#! /bin/bash

# Linux同步拷贝数据

# 服务端（被同步端）/etc/rsyncd.conf的内容：
# uid = root
# gid = root
# use chroot = yes
# transfer logging = true
# max connections = 10000
# log format = %h %o %f %l %b
# log file = /var/log/rsyncd.log
# pid file = /var/run/rsyncd.pid
# [rsyncd]
# path = /tmp/aaa
# ignore errors
# auth users = root
# secrets file = /etc/rsyncd.secrets

# 服务端（被同步端）/etc/rsyncd.secrets的内容：
# root:密码

# 服务端（被同步端）执行/etc/init.d/rsyncd restart


# 客户端（同步端）/etc/rsync.secret的内容：
# root:密码

# 客户端（同步端）全量模式：
rsync -qurtopg --process --delete root@服务端ip::rsyncd 目标路径 --password-file=/etc/rsync.secret

# 客户端（同步端）增量模式：
rsync -qurtopg --delete root@服务端ip::rsyncd 目标路径 --password-file=/etc/rsync.secret
