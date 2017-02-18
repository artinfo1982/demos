


#! /bin/bash

ip="$1"
port="$2"

# 建立链接
exec 6<>/dev/tcp/${ip}/${port}  # 如果是udp，则将tcp换成udp，在/proc/self/fd下面会生成软链接

# 发送数据
echo -e "abc" >&6

# 接收数据
cat <&6

# 关闭链接
exec 6>&-
