#! /bin/bash

#CPU
lscpu

#内存
dmidecode -t memory

#硬盘
lsblk

#网卡
lspci | grep "Eth"
