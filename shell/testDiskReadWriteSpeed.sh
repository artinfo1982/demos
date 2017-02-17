


#! /bin/bash

#磁盘写
dd if=/dev/zero of=a bs=8K count=256K

#磁盘读
hdparm -Tt /dev/sda
