# C++的若干技巧

## 模板
例如max函数，通常情况下可以这样定义：
```C++
int max(int x, int y) {
	return (x>y)?x:y;
}
float max(float x, float y) {
	return (x>y)?x:y;
}
double max(double x, double y) {
	return (x>y)?x:y;
}
```
但如果我们想使用char a,b; max(a, b);就会报错，因为我们没有定义char max(char x, char y)。能否只写一份代码就解决问题，可以使用模板。   
模板的一般形式如下，既可以是基本数据类型模板，也可以是类类型模板：
```text
template <class T> 或者 template <typename T>
返回类型 函数名（形参列表）
{函数体}
```
基本数据类型模板使用的示例：
```C++
#include <iostream>
using std::cout;
using std::endl;

template <class T> // 也可以是template <typename T>
T max(T x, T y) {
    return(x>y)?x:y;
}

int main( )
{
    int n1=2,n2=10;
    double d1=1.5,d2=5.6;
    cout<<max(n1,n2)<<endl;
    cout<<max(d1,d2)<<endl;
    return 0;
}
```
