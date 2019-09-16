/*
N进制小数，保留8位有效数字
gcc xx.c -o a -lm
*/

#include <stdio.h>
#include <math.h>

int main()
{
	float n, tmp;
	scanf("%f", &n);
	int i, j;
	int t;
	for (i = 2; i < 10; ++i)
	{
		tmp = n;
		printf("十进制数 %f 转为 %d 进制数: %d.", n, i, (int)floorf(n));
		for (j = 0; j < 8; ++j)
		{
			t = floorf(tmp * i);
			tmp = tmp * i - t;
			printf("%d", t);
		}
		printf("\n");
	}
	return 0;
}
