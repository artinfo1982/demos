/*
若有一只兔子每个月生一只小兔子，一个月后也开始生产。
起初只有一只兔子，一个月后就有两只兔子，二个月后就有三只兔子，
三个月后有五只兔子(小兔子投入生产)……
称之为费式数列，例如：1，1，2，3，5，8，13，21，34，55，89

解法：
F(n) = F(n-1) + F(n-2)
*/

#include <stdio.h>

unsigned long long fibonacci(int n)
{
    if (n == 1)
        return 1;
    if (n == 2)
        return 1;
    return fibonacci(n-1) + fibonacci(n-2);
}

int main()
{
    int i;
    for (i = 1; i <= 20; ++i)
        printf("%llu ", fibonacci(i));
    printf("\n");
    return 0;
}