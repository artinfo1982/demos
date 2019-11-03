#include <iostream>
#include <unordered_map>
#include <map>

using namespace std;

int main()
{
	int m, n;
	cin >> m >> n;
	int t;
	unordered_map<int, int> v;
	map<int, int> res;
	for (int i = 0; i < m; ++i) {
		cin >> t;
		v.insert(unordered_map<int, int>::value_type(t, i));
	}
	unordered_map<int, int>::iterator it1, it2, it3;
	for (it1 = v.begin(); it1 != v.end(); ++it1) {
		res.clear();
		int a = n - (*it1).first;
		it2 = it1;
		it2++;
		for (; it2 != v.end(); ++it2) {
			int b = a - (*it2).first;
			it3 = it2;
			it3++;
			for (; it3 != v.end(); ++it3) {
				auto j = v.find(b - (*it3).first);
				if (j != v.end()) {
					res.insert(map<int, int>::value_type((*it1).second, 0));
					res.insert(map<int, int>::value_type((*it2).second, 0));
					res.insert(map<int, int>::value_type((*it3).second, 0));
					res.insert(map<int, int>::value_type((*j).second, 0));
					v.erase(it1);
					v.erase(it2);
					v.erase(it3);
					v.erase(j);
				}
			}
		}
		for (auto k : res) {
			cout << k.first << " ";
		}
		cout << endl;
	}
	return 0;
}
