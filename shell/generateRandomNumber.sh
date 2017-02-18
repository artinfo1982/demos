


#! /bin/bash

# shell中生成指定范围的随机数

function rand() {
        local beg=$1
        local end=$2
        echo $((RANDOM % $end + $beg))
}

rand 1 10
