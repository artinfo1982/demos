#include <iostream>
#include <vector>

using namespace std;

#define MAX(x, y) (x)>(y)?(x):(y)

string func(string s)
{
	if (s.empty()) {
		return "";
	}
	int size = s.size();
	vector<pair<int, int>> v;
	int j = 0, k = 0;
	int max = 0;
	string res;
	for (int i = 0; i < size; ++i) {
		if (s[i] != '+' && s[i] != '-' && (s[i] < '0' || s[i] > '9')) {
			v.push_back(make_pair(i, 0));
		} else if (s[i] == '+' || s[i] == '-') {
			j = i+1;
			if (s[j] < '0' || s[j] > '9') {
				v.push_back(make_pair(i, 0));
				continue;
			}
			while (j < size) {
				if (s[j] != '+' && s[j] != '-' && s[j] != '.' && (s[j] < '0' || s[j] > '9')) {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '+' || s[j] == '-') {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '.' && j == size-1) {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '.' && (s[j+1] < '0' && s[j+1] > '9')) {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '.' && s[j+1] >= '0' && s[j+1] <= '9') {
					k = j+1;
					while (k < size && s[k] >= '0' && s[k] <= '9') {
						k++;
					}
					v.push_back(make_pair(i, k-i));
					break;
				} else {
					j++;
				}
			}
		} else if (s[i] == '.') {
			v.push_back(make_pair(i, 0));
		} else {
			j = i+1;
			while (j < size) {
				if (s[j] != '+' && s[j] != '-' && s[j] != '.' && (s[j] < '0' || s[j] > '9')) {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '+' || s[j] == '-') {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '.' && j == size-1) {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '.' && (s[j+1] < '0' && s[j+1] > '9')) {
					v.push_back(make_pair(i, j-i));
					break;
				} else if (s[j] == '.' && s[j+1] >= '0' && s[j+1] <= '9') {
					k = j+1;
					while (k < size && s[k] >= '0' && s[k] <= '9') {
						k++;
					}
					v.push_back(make_pair(i, k-i));
					break;
				} else {
					j++;
				}
			}
		}
	}

	for (int i = 0; i < v.size(); ++i) {
		max = MAX(max, v[i].second);
	}

	for (int i = 0; i < v.size(); ++i) {
		if (v[i].second == max) {
			res = s.substr(v[i].first, max);
		}
	}
	return res;
}

int main()
{
	string s = "1234567890abc+132456.26.9ed";
	cout << func(s) << endl;
	return 0;
}
