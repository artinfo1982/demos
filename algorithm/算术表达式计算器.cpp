/*
原始的字符串表示的算式中有很多多余的空格，请将其删除，并显示最终的计算结果。
例如：
原始字符串： 1 + 2     + 3
返回：6
*/

#include <iostream>
#include <string.h>
#include <vector>

using namespace std;

int main()
{
	char a[100];
	char b[100];
	cin.getline(a, 100);
	int i = 0, j = 0, k;
	memset(b, 0x0, 100);
	for (i = 0; i < strlen(a); ++i)
	{
		if (a[i] != ' ')
			b[j++] = a[i];
	}
	vector<int> x;
	vector<char> y;
	char c[8];
	int tmp = 0;
	for (i = 0; i < strlen(b); ++i)
	{
		for (j = i + 1; j <= strlen(b); ++j)
		{
			if (b[j] == '+' || b[j] == '-' || b[j] == '\0')
			{
				memset(c, 0x0, 8);
				for (k = 0; k < j-i; ++k)
					c[k] = b[i+k];
				x.push_back(atoi(c));
				y.push_back(b[j]);
				i = j + 1;
			}
		}
	}

	int sum = x[0];
	for (i = 0; i < y.size(); ++i)
	{
		if (y[i] == '+')
			sum += x[i + 1];
		else if (y[i] == '-')
			sum -= x[i + 1];
	}
	cout << sum << endl;

	return 0;
}
