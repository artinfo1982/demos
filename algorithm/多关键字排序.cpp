#include <iostream>
#include <stack>
#include <vector>
#include <map>

using namespace std;

int main()
{
	vector<stack<int>> v;
	map<int, stack<int>> x;
	map<int, stack<int>, greater<int>> y;
	int m, n, i, j, z, tmp;
	cin >> m >> n;
	int f[n] = {0};
	int res[m][n] = {0};
	for (i = 0; i < n; ++i)
	{
		cin >> z;
		f[i] = z;
	}
	for (i = 0; i < m; ++i)
	{
		stack<int> line;
		for (j = 0; j < n; ++j)
		{
			cin >> z;
			line.push(z);
		}
		v.push_back(line);
	}
	for (j = 0; j < n; ++j)
	{
		if (f[j] == 1)
		{
			for (i = 0; i < m; ++i)
			{
				tmp = v[i].top();
				v[i].pop();
				x.insert(map<int, stack<int>>::value_type(tmp, v[i]));
			}
			i = 0;
			for (auto s: x)
				res[i++][j] = s.first;
			x.clear();
		}
		else if (f[j] == -1)
		{
			for (i = 0; i < m; ++i)
			{
				tmp = v[i].top();
				v[i].pop();
				y.insert(map<int, stack<int>>::value_type(tmp, v[i]));
			}
			i = 0;
			for (auto s: y)
				res[i++][j] = s.first;
			y.clear();
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
