#include <iostream>
#include <list>

using namespace std;

bool cmp(const list<int> &x, const list<int> &y)
{
	return x.front() > y.front();
}

int main()
{
	int m, n, i, j, z, tmp;
	cin >> m >> n;
	int f[n] = {0};
	list<list<int> > a;
	int res[m][n] = {0};
	for (i = 0; i < n; ++i)
	{
		cin >> z;
		f[i] = z;
	}
	for (i = 0; i < m; ++i)
	{
		list<int> l;
		for (j = 0; j < n; ++j)
		{
			cin >> z;
			l.push_back(z);
		}
		a.push_back(l);
	}
	int idx = 0;
	list<list<int> >::iterator it;
	for (i = 0; i < n; ++i)
	{
		idx = 0;
		if (f[i] == 1)
			a.sort();
		else if (f[i] == -1)
			a.sort(cmp);
		for (it = a.begin(); it != a.end(); ++it)
		{
			res[idx++][i] = (*it).front();
			(*it).pop_front();
		}
	}
	for (i = 0; i < m; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			cout << res[i][j];
			if (j < n-1)
				cout << " ";
		}
		if (i < m-1)
			cout << endl;
	}
	return 0;
}
