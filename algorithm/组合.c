/*
利用位运算，求组合数
*/

#include <stdio.h>

unsigned long long nextN(int n)
{
    return (n+(n&(-n))) | ((n^(n+(n&(-n))))/(n&(-n)))>>2;
}

// 从n个元素中选取m个元素的组合
unsigned long long combination(int n, int m)
{
    unsigned long long c = (1<<m)-1;
    unsigned long long result = 0;
    while(c<=((1<<n)-(1<<(n-m))))
    {
        c = nextN(c);
        result++;
    }
    return result;
}

int main()
{
    int n, m;
    while (scanf("%d %d", &n, &m) != EOF)
    {
        printf("%llu\n", combination(n, m));
    }
    return 0;
}