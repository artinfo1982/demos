/*
将一个字符串中除了空格外的所有特殊字符都删除，每两个单词之间只能有一个空格。
每个单词首字母大写，在句子的最后是一个英文的点号。

例如：
原始字符串：who *juh (MOLO)
最终字符串：Who Juh Molo.
*/

#include <iostream>
#include <string.h>

using namespace std;

int main()
{
	char c[100];
	memset(c, 0x0, 100);
	cin.getline(c, 100);
	int len = strlen(c);
	for (int i = 0; i < len; ++i)
	{
		if (i == 0 && c[i] >= 97 && c[i] <= 122)
			c[i] -= 32;
		else if ((c[i] < 65) || (c[i] > 90 && c[i] < 97) || (c[i] > 122))
			c[i] = ' ';
	}
	for (int i = 1; i < len; ++i)
	{
		if (c[i] >= 97 && c[i] <= 122 && c[i-1] == ' ')
			c[i] -= 32;
	}
	bool flag = false;
	for (int i = 0; i < len; ++i)
	{
		if (c[i] != ' ')
		{
			flag = true;
			cout << c[i];
		}
		else if (c[i] == ' ' && c[i+1] == ' ')
			continue;
		else if (c[i] == ' ' && c[i+1] != ' ' && flag)
			cout << ' ';
	}
	cout << '.' << endl;
	return 0;
}
