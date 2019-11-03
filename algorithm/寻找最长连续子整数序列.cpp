#include <iostream>
#include <unordered_map>

#define MAX(x, y) (x) > (y) ? (x) : (y)

using namespace std;

int main()
{
	int n;
	cin >> n;
	int t;
	unordered_map<int, bool> m;
	for (int i = 0; i < n; ++i) {
		cin >> t;
		m.insert(unordered_map<int, bool>::value_type(t, false));
	}
	int res = 0, tmp = 0;
	unordered_map<int, bool>::iterator it;
	for (it = m.begin(); it != m.end(); ++it) {
		if ((*it).second) {
			continue;
		}
		tmp = 0;
		(*it).second = true;
		int key1 = (*it).first - 1;
		int key2 = (*it).first + 1;
		if (m.find(key1) != m.end() || m.find(key2) != m.end()) {
			tmp++;
		}
		while (m.find(key1) != m.end() && !m[key1]) {
			m[key1] = true;
			key1--;
			tmp++;
		}
		while (m.find(key2) != m.end() && !m[key2]) {
			m[key2] = true;
			key2++;
			tmp++;
		}
		res = MAX(res, tmp);
	}
	cout << res << endl;
	return 0;
}
