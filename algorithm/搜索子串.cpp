#include <iostream>

using namespace std;

bool kmp(char* a, char* b)
{
	if (a == NULL || b == NULL)
		return false;
	char *p1 = a;
	char *p2 = b;
	char *p3 = NULL;
	char *p4 = NULL;

	while(*p1 != '\0')
	{
		if (*p1 == *p2)
		{
			p3 = p1;
			p4 = p2;
			while(*p4 != '\0')
			{
				if (*p4 != *p3)
				{
					p1 = p3;
					break;
				}
				else
				{
					p3++;
					p4++;
				}
			}
			if (*p4 == '\0')
				return true;
		}
		else
			p1++;
	}
}

int main()
{
	char *a = (char*)"fuckyou";
	char *b = (char*)"you";
	char *c = (char*)"cky";
	char *d = (char*)"ucdy";

	cout << kmp(a, b) << endl;
	cout << kmp(a, c) << endl;
	cout << kmp(a, d) << endl;

	return 0;
}
