# Shell脚本的若干有用例子
## Linux主机之间实现自动ssh访问
```shell
#! /usr/bin/expect
set ip [lindex $argv 0]
set passwd [lindex $argv 1]
spawn ssh root@${ip} "date"
expect {
  "(yes/no)?" {
    send "yes\r"
    expect "assword:"
    send "${passwd}\r"
  }
  "assword:" {send "${passwd}\r"}
  "* to host" {exit 0}
}
expect eof
```
## 从一个string的指定位置开始截取指定个数的字符
```shell
#! /bin/bash
function cutChars() {
        #输入字符串
        local s=$1
        #从哪一位字符开始，0表示第一个字符
        local beg=$2
        #向后截取几个字符
        local num=$3
        echo "${s:$beg:$num}"
}
#输出bcd
cutChars "abcdefg" 1 3
```
## 模拟吃CPU
```shell
#! /bin/bash
# 使用方法：./eatCpu.sh 期望吃掉的CPU核数
for i in `seq $1`
do
  echo -ne "
  i=0;
  while true
  do
    i=i+1;
  done" | /bin/sh &
  pid_array[$i]=$!
done
```
## 生成指定范围内的随机整数
```shell
#! /bin/bash
function rand() {
        local beg=$1
        local end=$2
        echo $((RANDOM % $end + $beg))
}
#生成1-10之间的随机整数
rand 1 10
```
## CPU使用率
```shell
#! /bin/bash
mpstat 1 1
```
## 获取当前时间
```shell
#! /bin/bash
#精确到秒
date +%s
#精确到纳秒
date +%s.%N
#格式为 2018-01-01 10:10:10.245
date +"%Y-%m-%d %H:%M:%S.%N" | cut -c1-23
```
## 磁盘读写速率
```shell
#! /bin/bash
iostat -d -x -k 1 2
```
## 显示硬件信息
```shell
#! /bin/bash
#CPU
lscpu
#内存
dmidecode -t memory
#硬盘
lsblk
#网卡
lspci | grep "Eth"
#显卡
lspci | grep "3D"
```
## 网络流量统计
```shell
#! /bin/bash
sar -n DEV 1 1
```
## 使用rsync同步数据
```shell
#! /bin/bash
#################################################
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
#################################################

# 客户端（同步端）全量模式：
rsync -qurtopg --process --delete root@服务端ip::rsyncd 目标路径 --password-file=/etc/rsync.secret

# 客户端（同步端）增量模式：
rsync -qurtopg --delete root@服务端ip::rsyncd 目标路径 --password-file=/etc/rsync.secret
```
## shell使用switch-case示例
```shell
#! /bin/bash

function A()
{
	local input="$1"
	if [ "${input}" == "1" ]; then
		echo "one"
		exit 0
	else
		echo "error"
		exit 1
	fi
}

function B()
{
	local input="$1"
	if [ "${input}" == "2" ]; then
                echo "two"
                exit 0
        else
                echo "error"
                exit 1
        fi
}

input_1="$1"
input_2="$2"

if [ $# -ne 2 ]; then
	echo "input parmas error"
	echo "e.g. $0 input_1 input_2"
	exit 1
fi

case ${input_1} in
	1) A ${input_2} ;;
	2) B ${input_2} ;;
	*) echo "INVALID NUMBER!" ;;
esac
```
## 使用tc模拟网络损伤
```shell
#! /bin/bash
# 时延
tc qdisc add dev eth0 root netem delay 1000ms
# 丢包10%
tc qdisc add dev eth0 root netem loss 10%
# 删除tc记录
tc qdisc del dev eth0 root
# 查看tc记录
tc qdisc show
```
## shell实现tcp、udp通信
```shell
#! /bin/bash
ip="$1"
port="$2"
# 建立链接，如果是udp，则将tcp换成udp，在/proc/self/fd下面会生成软链接
exec 6<>/dev/tcp/${ip}/${port}
# 发送数据
echo -e "abc" >&6
# 接收数据
cat <&6
# 关闭链接
exec 6>&-
```
## 磁盘读写性能测试
```shell
#! /bin/bash
#磁盘写
dd if=/dev/zero of=a bs=8K count=256K
#磁盘读
hdparm -Tt /dev/sda
```
## tmpfs
```shell
#! /bin/bash
mount tmpfs /opt/tmpfs -t tmpfs -o size=32G
```
## 使用python解析curl返回的json
```shell
#! /bin/bash
#假设curl返回的json结构为：{"data":"123"}
curl -s "http://192.168.1.1:50000/a" | python -c "import json,sys;obj=json.load(sys.stdin);print obj['data']"
```
## 生成自增序列，左补零
```shell
#例如下例，会生成 001 002 ... 100
seq -w 1 100
```
