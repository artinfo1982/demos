#include <stdio.h>
#include <stdlib.h>

#define MAX 100000

int main()
{
    int n;
    while (scanf("%d", &n) != EOF)
    {
        if (n < 0 || n >= 10000)
            break;
        if (0 == n || 1 == n)
        {
            printf("1\n");
            break;
        }

        int i;
        unsigned int *p = (unsigned int *)malloc(MAX * sizeof(unsigned int));
        for (i = 0; i < MAX; ++i)
            *(p + i) = 0;
        *(p + 1) = 1; // 从p[1]开始
        int point = 1; // point表示位数，刚开始只有一位p[1] 且p[1] = 1，不能为0，0乘任何数为0
        int carry = 0; // carry表示进位数，刚开始进位为0
        int j = 0;
        int temp;
        for (i = 2; i <= n; ++i) // n的阶乘
        {
            for (j = 1; j <= point; ++j) // 循环p[]，让每一位都与i乘
            {
                temp = *(p + j) * i + carry; // temp变量表示不考虑进位的值
                carry = temp / 10; // 计算进位大小
                *(p + j) = temp % 10; // 计算本位值
            }
            // 处理最后一位的进位情况
            // 由于计算数组的最后一位也得考虑进位情况，所以用循环讨论
            // 因为可能最后一位可以进多位；比如 12 * 本位数8，可以进两位
            while(carry) // 当进位数存在时，循环的作用就是将一个数分割，分割的每一位放入数组中
            {
                *(p + j) = carry % 10;
                carry = carry / 10;
                j++; // 表示下一位
            }
            point = j - 1; // 由于上面while中循环有j++,所以位会多出一位，这里减去
        }
        for (i = point; i >= 1; --i) // 逆序打印结果
            printf("%d", *(p + i));
        printf("\n");
    }
    return 0;
}