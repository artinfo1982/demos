#include <iostream>

using namespace std;

int main()
{
	int m, n, i, j, k;
	int sum = 0;
	cin >> m >> n;
	int a[m][m];
	int col[m]; // 列向量存储入度
	int row[m]; // 行向量存储出度
	int idx = 0;
	string s;
	for (i = 0; i < m; ++i)
	{
		col[i] = 0;
		row[i] = 0;
		for (j = 0; j < m; ++j)
			a[i][j] = 0;
	}
	bool flag1 = false; // 判定是否有矛盾
	bool flag2 = true; // 判定入度数组中是否还有0
	for (k = 0; k < n; ++k)
	{
		string line;
		cin >> line;
		int x = line.at(0) - 65;
		int y = line.at(2) - 65;
		a[x][y] = 1;
		// 计算所有点的入度，并存入数组
		for (j = 0; j < m; ++j)
		{
			sum = 0;
			for (i = 0; i < m; ++i)
				sum += a[i][j];
			col[j] = sum;
		}
		// 计算所有点的出度，并存入数组
		for (i = 0; i < m; ++i)
		{
			sum = 0;
			for (j = 0; j < m; ++j)
				sum += a[i][j];
			row[i] = sum;
		}
		// 如果所有入度都不是0，说明有矛盾的情况。
		// 如果有入度为0，但对应的出度也为0，说明要么输入还没有结束，要么是孤立点，
		// 需要判断其余列是否有矛盾
		flag1 = false;
		for (i = 0; i < m; ++i)
		{
			if ((col[i] == 0) && (row[i] == 0))
				continue;
			else if (col[i] == 0)
			{
				flag1 = true;
				break;
			}
		}
		if (!flag1)
		{
			cout << "Inconsistency found after " << k+1 << " relations." << endl;
			return 0;
		}
		// 如果入度数组中只有唯一的0，说明存在解的可能性
		sum = 0;
		for (i = 0; i < m; ++i)	
		{
			if (col[i] == 0)
				sum++;
		}
		if (sum == 1)
		{
			while (flag2)
			{
				for (i = 0; i < m; ++i)
				{
					if (col[i] == 0)
					{
						char r = i + 65;
						s.push_back(r);
						idx++;
						for (j = 0; j < m; ++j)
							a[i][j] = 0;
						a[i][i] = 1;
						break;
					}
				}
				for (j = 0; j < m; ++j)
        		{
            		sum = 0;
            		for (i = 0; i < m; ++i)
                		sum += a[i][j];
            		col[j] = sum;
        		}
				sum = 0;
				for (i = 0; i < m; ++i)
				{
					if (col[i] == 0)
						sum++;
				}
				if (sum != 1)
					flag2 = false;
			}
		}
		if (idx == m)
		{
			cout << "Sorted sequence determined after " << k+1 << " relations: " << s << endl;
			return 0;
		}
	}
	if (idx < m)
		cout << "Sorted sequence cannot be determined." << endl;
	return 0;
}
