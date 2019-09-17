/*
输入一个正整数，1<=N<=10^9，找出其所有因数的排列。
输出的第一个数字为因数的个数，后面是自小到大的因数序列。

示例：
输入：
12
输出：
1 2 3 4 6 12
*/

#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

bool isPrime(long n)
{
	if (n <= 3)
		return n > 1;
	if (n % 6 != 1 && n % 6 != 5)
		return false;
	int s = sqrt(n);
	int i;
	for (i = 5; i <= s; i += 6)
	{
		if (n % i == 0 || n % (i + 2) == 0)
			return false;
	}
	return true;
}

int main()
{
	long n;
	cin>>n;
	long sqr;
	long i;
	vector<string> v1;
	vector<string> v2;
	if (n == 1)
		printf("1 1\n");
	else if (isPrime(n))
		printf("2 1 %ld\n", n);
	else
	{
		sqr = floor(sqrt(n));
		for (i = 2; i <= sqr; ++i)
		{
			if (n % i == 0)
				v1.push_back(to_string(i));
		}
		for (i = v1.size()-1; i >=0; --i)
			v2.push_back(to_string(n / stol(v1[i])));
		printf("%ld 1 ", v1.size() + v2.size() + 2);
		for (auto v : v1)
			cout << v << " ";
		for (auto v : v2)
			cout << v << " ";
		printf("%ld\n", n);
	}
	return 0;
}
