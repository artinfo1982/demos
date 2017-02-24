


#! /bin/bash

# 模拟吃CPU资源，吃掉一个核就是吃掉100%
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

for i in "${pid_array[@]}"
do
  echo "kill $i"
done
