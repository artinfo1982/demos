


#! /bin/bash

# 从一个string的制定位置开始截取指定个数个字符

function cutChars() {
        local s=$1  #字符串
        local beg=$2   #从哪一位字符开始，0表示第一个字符
        local num=$3   #向后截取几个字符
        echo "${s:$beg:$num}"
}

cutChars "abcdefg" 1 3  #输出bcd
