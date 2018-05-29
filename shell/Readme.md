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
## 从一个string的制定位置开始截取指定个数的字符
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
