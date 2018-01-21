#include <stdlib.h>
#include <iostream>

using namespace std;

/**
* 写一个程序，接受一个以N/D的形式输入的分数，其中N为分子，D为分母，输出它的小数形式。
* 如果它的小数形式存在循环节，要将其用括号括起来。
* 比如1/3 = 0.(3)，1/7 = 0.(142857)
*/
void decimal(const int n, const int d)
{
	int a = n;
	int b = d;
	int t;
	int used[10000];
	memset(used, 0, sizeof(used));
	a %= b;
	//能循环就是出现余数重复出现,则只需找到第一次重复出现的余数
	//0是结束标志,先将a置为重复出现,然后每次a = a*10%b求出新得的余数,并检查该余数是否出现过,有则开始重复,无则置为1,继续
	for(used[0] = 1, used[a] = 1, a = a*10%b; used[a] != 1 ; used[a] = 1, a = a*10%b)
	{
		;
	}
	if(a == 0)
	{
		printf("%d/%d : %g\n", n, d, (double)n/d);
	}
	else
	{
		t = a;
		printf("%d/%d : 0.(", a, b);
		do 
		{
			printf("%d", a*10/b);
			a = a*10%b;
		} while (a != t);
		printf(")\n");
	}
}

int main()
{
	decimal(22, 4);
	decimal(1, 3);
	decimal(1, 7);
	return 0;
}
