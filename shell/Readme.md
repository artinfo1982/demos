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
## 从一个string的制定位置开始截取指定个数个字符
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
