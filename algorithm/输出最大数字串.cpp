#include <iostream>
#include <map>

using namespace std;

int main() {
	string s;
	cin >> s;
	map<int, string> m;
	for (int i = 0; i < s.size(); ++i) {
		char c1 = s.at(i);
		if (c1 != '+' && c1 != '-' && c1 != '.' && (c1 < '0' || c1 > '9')) {
			s[i] = ' ';
		}
		else if (c1 == '+' || c1 == '-') {
			if (i == 0 || i == s.size()-1) {
				s[i] = ' ';
			}
			int j = 1;
			while (i+j < s.size() && (s.at(i+j) < '0' || s.at(i+j) > '9')) {
				j++;
			}
			for (int k = i; k < i+j-1; ++k) {
				s[k] = ' ';
			}
		}
		else if (c1 == '.') {
			if (i == 0 || i == s.size()-1) {
				s[i] = ' ';
			}
			else if (s.at(i-1) < '0' || s.at(i-1) > '9') {
				s[i] = ' ';
			}
			int m = 1;
			while (i+m < s.size() && (s.at(i+m) < '0' || s.at(i+m) > '9')) {
				m++;
			}
			for (int k = i; k < i+m-1; ++k) {
				s[k] = ' ';
			}
		}
	}

	for (int i = 0; i < s.size(); ++i) {
		if (s.at(i) == ' ') {
			continue;
		}
		int j = 1;
		while (i+j < s.size() && s.at(i+j) != ' ') {
			j++;
		}
		string tmp = s.substr(i, j);
		if (tmp.size() > 1 && tmp.at(0) == '0') {
			if (tmp.at(1) != '.') {
				while (tmp.at(0) == '0') {
					tmp.erase(0, 1);
				}
			}
		}
		m.insert(map<int, string>::value_type(tmp.size(), tmp));
		i += j;
	}

	map<int, string>::iterator it;
	it = m.end();
	it--;
	cout << (*it).second << endl;
	return 0;
}
