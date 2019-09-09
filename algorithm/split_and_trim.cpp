#include <vector>
#include <string>
#include <iostream>

using namespace std;
 
std::vector<std::string> split(const string &s, const string &seperator)
{
    vector<string> result;
    typedef string::size_type string_size;
    string_size i = 0;
     
    while(i != s.size())
    {
        //找到字符串中首个不等于分隔符的字母；
        int flag = 0;
        while(i != s.size() && flag == 0)
        {
            flag = 1;
            for(string_size x = 0; x < seperator.size(); ++x)
            {
                if(s[i] == seperator[x])
                {
                    ++i;
                    flag = 0;
                    break;
                }
            }
        }
         
        //找到又一个分隔符，将两个分隔符之间的字符串取出；
        flag = 0;
        string_size j = i;
        while(j != s.size() && flag == 0)
        {
            for(string_size x = 0; x < seperator.size(); ++x)
            {
                if(s[j] == seperator[x])
                {
                    flag = 1;
                    break;
                }
                if(flag == 0)
                    ++j;
            }
        }
        if(i != j)
        {
            result.push_back(s.substr(i, j-i));
            i = j;
        }
    }
    return result;
}

std::string& trim(std::string &s) 
{
    if (s.empty()) 
    {
        return s;
    }
    s.erase(0, s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
    return s;
}
 
int main()
{
    std::string s1 = "abc 123 qaz";
    std::vector<std::string> v = split(s1, " ");
    for(std::vector<std::string>::size_type i = 0; i != v.size(); ++i)
        cout << v[i] << " ";
    cout << endl;

    std::string s2 = " abc ";
    cout << trim(s2) << endl;
    
    return 0;
}