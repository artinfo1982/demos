#include <stdio.h>

/*
* 计算任意整数的二进制表达中1的个数
* 将该数不断与1求与，然后右移1位，直至所有的比特位都是0
*/

int main(int argc, char *argv[])
{
    int a = atoi(argv[1]);

    int b, i, bitMax, n=0;

    b = a;
    bitMax = sizeof(int) * 8;

    for (i=0; i<bitMax; i++)
    {
        if (0 == b)
            break;
        if (b & 0x01)
            n++;
        b >>= 1;
    }
    printf("input number is: %d, contains %d 1\n", a, n);
    return 0;
}
