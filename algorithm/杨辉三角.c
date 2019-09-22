/*
也叫帕斯卡三角形，是二项式系数的图形化表现。

    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1

杨辉三角的主要性质：
1.杨辉三角的值，对应组合的值，也就是C(2n,0), C(2n,1), ... C(2n,2n)
2.每一行的和 = 2^n
3.C(2n,1), C(2n,3), ... C(2n,2n-1)的最大公约数：如果n是2的幂，则最大公约数是2n，否则是2
4.C(2n,1) + C(2n,3) + ... + C(2n,2n-1) = C(2n,0) + C(2n,2) + ... + C(2n,2n) = 2^(n-1)
*/

#include <stdio.h>

unsigned long long func(int n, int idx)
{
    if (n == 1)
        return 1;
    if (n == 2)
        return 1;
    if ((idx == 1) || (idx == n))
        return 1;
    else
        return func(n-1, idx-1) + func(n-1, idx);
}

int main()
{
    int i, j, n = 20;
    for (i = 1; i <= n; ++i)
    {
        for (j = 1; j <= i; ++j)
            printf("%llu ", func(i, j));
        printf("\n");
    }
    return 0;
}
