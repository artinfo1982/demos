


#! /bin/bash

# 模拟网络时延、丢包

# 时延
tc qdisc add dev eth0 root netem delay 1000ms

# 丢包10%
tc qdisc add dev eth0 root netem loss 10%

# 删除tc记录
tc qdisc del dev eth0 root

# 查看tc记录
tc qdisc show
