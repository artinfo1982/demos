#! /bin/bash

function A()
{
	local input="$1"
	if [ "${input}" == "1" ]; then
		echo -e "\033[32mone\033[0m"
		exit 0
	else
		echo -e "\033[31merror\033[0m"
		exit 1
	fi
}

function B()
{
	local input="$1"
	if [ "${input}" == "2" ]; then
                echo -e "\033[32mtwo\033[0m"
                exit 0
        else
                echo -e "\033[31merror\033[0m"
                exit 1
        fi
}

function C()
{
	local input="$1"
	if [ "${input}" == "3" ]; then
                echo -e "\033[32mthree\033[0m"
                exit 0
        else
                echo -e "\033[31merror\033[0m"
                exit 1
        fi
}

input_1="$1"
input_2="$2"

if [ $# -ne 2 ]; then
	echo -e "\033[31minput parmas error\033[0m"
	echo -e "\033[31me.g. $0 input_1 input_2\033[0m"
	exit 1
fi

case ${input_1} in
	1) A ${input_2} ;;
	2) B ${input_2} ;;
	3) C ${input_2} ;;
	*) echo -e "\033[31mINVALID NUMBER!\033[0m" ;;
esac

exit 0
