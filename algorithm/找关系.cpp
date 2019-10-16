#include <iostream>
#include <map>

using namespace std;

int main()
{
	int m, n, i, j, k, mul = 1, sum = 0, x, y;
	cin >> m >> n;
	int a[m] = {0};
	map<int, int> res;
	int flag = (1+m)*m/2;
	string line;
	char c;
	for (i = 0; i < n; ++i)
	{
		cin >> line;
		x = line.at(0)-65;
		y = line.at(2)-65;
		if (a[x] == 0)
			a[x] = 1;
		else if (a[y] == 0)
			a[y] = 1;
		else if (a[x] > a[y])
		{
			cout << "Inconsistency found after " << i+1 << " relations." << endl;
			return 0;
		}
		a[y] = a[x] + 1;
		mul = 1;
		sum = 0;
		for (j = 0; j < m; ++j)
		{
			sum += a[j];
			mul *= a[j];
		}
		if (mul != 0 && sum == flag)
		{
			cout << "Sorted sequence determined after " << i+1 << " relations: ";
			res.clear();
			for (k = 0; k < m; ++k)
				res.insert(map<int, int>::value_type(a[k], k));
			for (auto w : res)
			{
				c = w.second + 65;
				cout << c;
			}
			cout << endl;
			return 0;
		}
	}

	for (i = 0; i < m; ++i)
	{
		if (a[i] == 0)
		{
			cout << "Sorted sequence cannot be determined." << endl;
			return 0;
		}
	}
	cout << "Sorted sequence determined after " << m << " relations: ";
	res.clear();
	for (k = 0; k < m; ++k)
		res.insert(map<int, int>::value_type(a[k], k));
	for (auto w : res)
	{
		c = w.second + 65;
		cout << c;
	}
	cout << endl;

	return 0;
}
