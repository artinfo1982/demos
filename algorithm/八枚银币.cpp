/*
现有八枚银币a b c d e f g h,已知其中一枚是假币,其重量不同于真币,但不知是较轻或
较重,如何使用天平以最少的比较次数,决定出哪枚是假币,并得知假币比真币较轻或较重
*/

#include <iostream>

using namespace std;

void print(int x, int y, int idx, int& num)
{
	if (x < y)
		printf("fake: #%d, less than\n", idx);
	else
		printf("fake: #%d, greater than\n", idx);
	num++;
}

void check(int a[])
{
	int num = 0; //总的比较次数
	// 假币必定在a[6]、a[7]中
    if (a[0] + a[1] + a[2] == a[3] + a[4] + a[5])
    {   
		num++;
        // a[7]是假币
        if (a[6] == a[0])
			print(a[7], a[0], 8, num);
		// a[6]是假币
		else
			print(a[6], a[0], 7, num);
    }
	// 假币必定在a[4]、a[5]中
	else if (a[0] + a[1] == a[2] + a[3])
	{
		num++;
		// a[5]是假币
		if (a[4] == a[6])
			print(a[5], a[6], 6, num);
		// a[4]是假币
		else
			print(a[4], a[6], 5, num);
	}
	// 假币必定在a[2]、a[3]中
	else if (a[0] == a[1])
	{
		num++;
		if (a[2] == a[6])
			print(a[3], a[6], 4, num);
		else
			print(a[2], a[6], 3, num);
	}
	// 假币必定在a[0]、a[1]中
	else
	{
		if (a[0] == a[6])
			print(a[1], a[6], 2, num);
		else
			print(a[0], a[6], 1, num);
	}
	printf("total use cmp: %d\n", num);
}

int main()
{
	int a[8] = {2, 2, 2, 2, 1, 2, 2, 2};
	int b[8] = {2, 3, 2, 2, 2, 2, 2, 2};
	check(a);
	printf("\n");
	check(b);
	printf("\n");
	return 0;
}
