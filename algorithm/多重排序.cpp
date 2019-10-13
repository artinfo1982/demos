#include <iostream>
#include <stdlib.h>

using namespace std;

int cmp1(const void *a, const void *b)
{
	return *(int*)a - *(int*)b;
}

int cmp2(const void *a, const void *b) 
{
    return *(int*)b - *(int*)a;
}

int main()
{
	int m, n, i, j, z, tmp;
	cin >> m >> n;
	int f[n] = {0};
	int a[n][m] = {0};
	for (i = 0; i < n; ++i)
	{
		cin >> z;
		f[i] = z;
	}
	for (j = 0; j < n; ++j)
	{
		for (i = 0; i < m; ++i)
		{
			cin >> z;
			a[i][j] = z;
		}
	}
	for (j = n-1; j >= 0; --j)
	{
		if (f[j] == 1)
			qsort(a[j], m, sizeof(a[j][0]), cmp1);
		else if (f[j] == -1)
			qsort(a[j], m, sizeof(a[j][0]), cmp2);
	}
	for (j = 0; j < m; ++j)
	{
		for (i = 0; i < n; ++i)
		{
			cout << a[i][j];
			if (i < n-1)
				cout << " ";
		}
		if (j < m-1)
			cout << endl;
	}
	return 0;
}
