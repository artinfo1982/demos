#include <iostream>

#define N 10

using namespace std;

int main()
{
	int n;
	int h[N];
	h[0] = 1;
	h[1] = 1;
	for (int i = 2; i < N; ++i)
		h[i] = 0;

	for (int i = 2; i < N; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			h[i] += h[j] * h[i-1-j];
		}
	}

	for (int i = 0; i < 10; ++i)
		cout << h[i] << endl;

	return 0;
}
