/*
* 判断一个路径是文件还是文件夹
*/

#include <iostream>
#include <sys/stat.h>

using namespace std;

int main()
{
    struct stat buf;
    if (lstat("/home/cd/tmp", &buf) < 0)
    {
        cout << "lstat error" << endl;
        return 1;
    }
    if (S_ISDIR(buf.mode))
        cout << "is dir" << endl;
    if (S_ISREG(buf.mode))
        cout << "is file" << endl;
    return 0;
}