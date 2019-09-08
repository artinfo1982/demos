#include <stdlib.h>
#include <iostream>

using namespace std;

/**
* 写一个程序，接受一个字符串，然后输出转换为小写之后的字符串，不在字母范围内的字符，需丢弃。
* 0表示处理成功，-1表示失败。
*/

// ascii码表，A~Z：65~90，a~z：97~122，每个大写字母加32就是对应的小写字母
int process(const char * strInput, char * strOutput)
{
	if (NULL == strInput)
		return -1;
	int len = 0, i = 0, j = 0;
	while(*(strInput + i) != '\0')
	{
		len++;
		i++;
	}
	for (i = 0; i < len; i++)
	{
		//是大写
		if ((*(strInput + i) >= 65) && (*(strInput + i) <= 90))
		{
			*(strOutput + j) = *(strInput + i) + 32;
			j++;
		}
		//是小写
		else if ((*(strInput + i) >= 97) && (*(strInput + i) <= 122))
		{
			*(strOutput + j) = *(strInput + i);
			j++;
		}
		else
			continue;
	}
	*(strOutput + j) = '\0';
	return 0;
}

int main()
{
	char * str = "AjU4mG0*gS#";
	char * p = (char *)malloc(32 * sizeof(char));
	if (0 == process(str, p))
		cout << p << endl;
	else
		cout << "error" << endl;
	free(p);
	p=NULL;
	return 0;
}
