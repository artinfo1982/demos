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
但如果我们想使用char a,b; max(a,b);就会报错，因为我们没有定义char max(char x, char y)。能否只写一份代码就解决问题，可以使用模板。   
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
类类型的模板一般的定义形式：
```text
template <class T> 或者 template <typename T>
class 类名 {
//类定义
};
```
```C++
// ClassTemplateDemo.hpp
#ifndef CLASS_TEMPLATE_DEMO_HH
#define CLASS_TEMPLATE_DEMO_HH

template<typename T1, typename T2> //模板参数可以是一个，也可以是多个
class ClassTemplateDemo{
private:
    T1 m;
    T2 n;
public:
    ClassTemplateDemo(T1 a, T2 b); //构造函数
    void show();
};

template <typename T1, typename T2>
ClassTemplateDemo<T1, T2>::ClassTemplateDemo(T1 a, T2 b):m(a), n(b){}

template <typename T1, typename T2>
void ClassTemplateDemo<T1,T2>::show()
{
    cout<<"m="<<m<<", n="<<n<<endl;
}
#endif


// ClassTemplateDemo.cpp
#include <iostream>
#include "ClassTemplateDemo.hpp"
using std::cout;
using std::endl;

int main()
{
    ClassTemplateDemo<int, int> class1(3, 5);
    class1.show();
    ClassTemplateDemo<int, char> class2(3, 'a');
    class2.show();
    ClassTemplateDemo<double, int> class3(2.9, 10);
    class3.show();
    return 0;
}
```
