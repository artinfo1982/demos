#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    int n;
    int m;
    int i, j;
    int num;

    while (~scanf("%d", &n))
    {
        if (n == 2)
        {
            printf("0\n");
            continue;
        }

        if (n % 2 == 1)
            m = n / 2 + 1;
        else
            m = n / 2;
        
        char *p = (char*)malloc(m * sizeof(char));
        memset(p, 0x1, m);

        for (i = 3; i <= n; i=i+2)
        {
            for (j = 3; j*i < n; j=j+2)
                *(p + (i*j-1)/2) = 0x0;
        }

        num = 0;
        for (i = 1; i <= m; ++i)
        {
            if (i+1 < m && *(p+i) == 0x1 && *(p+i+1) == 0x1)
                num++;
        }
        printf("%d\n", num);
    }
    return 0;
}