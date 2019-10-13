#include <iostream>

using namespace std;

#define MIN(x, y) (x) < (y) ? (x) : (y)

int main()
{
	int n, i, j, z;
	cin >> n;
	int r[n][n+1] = {0};
	int dp[n+1] = {0};
	for (i = 0; i < n; ++i)
	{
		for (j = i+1; j <= n; ++j)
		{
			cin >> z;
			r[i][j] = z;
		}
	}
	dp[0] = 0;
	dp[1] = r[0][1];
	int tmp = 0;
	for (i = 2; i <= n; ++i)
	{
		tmp = 999999;
		for (j = 0; j < i; ++j)
			tmp = MIN(tmp, dp[j]+r[j][i]);
		dp[i] = tmp;
	}
	cout << dp[n] << endl;
	return 0;
}
