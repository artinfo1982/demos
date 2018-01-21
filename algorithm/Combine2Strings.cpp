#include <stdlib.h>
#include <iostream>

using namespace std;

/**
* 字符串合并，写一个程序，接受两个字符串，然后输出合并之后的字符串
*/
void combine(const char * str1, const char * str2)
{
	int len1=0, len2=0, i=0, j=0;
	while(*(str1 + i) != '\0')
	{
		len1++;
		i++;	
	}
	i=0;
	while(*(str2 + i) != '\0')
	{
		len2++;
		i++;	
	}
	char * p = (char *)malloc((len1+len2 + 1) * sizeof(char));
	for (i=0; i<len1; i++)
	{
		*(p + j) = *(str1 + i);
		j++;
	}
	for (i=0; i<len2; i++)
	{
		*(p + j) = *(str2 + i);
		j++;
	}
	*(p + j)='\0';
	cout << p << endl;
	free(p);
	p = NULL;
}

int main()
{
	char * str1 = "ABC";
	char * str2 = "def";
	combine(str1, str2);
	return 0;
}
