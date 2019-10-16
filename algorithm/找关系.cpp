#include <iostream>
#include <map>

using namespace std;

int main()
{
	int m, n, i, j, sum = 0, x, y;
	cin >> m >> n;
	int a[m] = {0};
	int b[m] = {0};
	map<char, int> s;
	int flag = (1+m)*m/2-m;
	string line;
	for (i = 0; i < n; ++i)
	{
		cin >> line;
		x = line.at(0)-65;
		y = line.at(2)-65;
		b[x] = 1;
		b[y] = 1;
		cout << "a[x]=" << a[x] << ", a[y]=" << a[y] << ", b[x]=" << b[x] << ", b[y]=" << b[y] << endl;
		if (a[y] < a[x] && b[x] == 1 && b[y] == 1)
		{
			cout << "Inconsistency found after " << i+1 << " relations." << endl;
			return 0;
		}
		a[y] = a[x] + 1;
		s.insert(map<char, int>::value_type(line.at(0), 0));
		s.insert(map<char, int>::value_type(line.at(2), 0));
		sum = 0;
		for (j = 0; j < m; ++j)
		{
			sum += a[j];
		}
		if (sum == flag)
		{
			cout << "Sorted sequence determined after " << i+1 << " relations: ";
			for (auto k : s)
				cout << k.first;
			cout << endl;
			return 0;
		}
	}

	for (i = 0; i < m; ++i)
	{
		if (a[i] == 0)
			sum++;
	}
	if (sum > 1)
	{
		cout << "Sorted sequence cannot be determined." << endl;
		return 0;
	}

	return 0;
}
