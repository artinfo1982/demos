# C++的若干经验技巧

## 常量表达式constexpr(c++11)
觉得一个变量是常量表达式，就把它定义为constexpr，交给编译器判断。
例如 constexpr int a = size();

## typedef
```C++
typedef double wage; // wage是double的别名
typedef char *p; // p是char*的别名
```

## using(c++11)
```C++
using SI = Week; // SI是Week的别名
Si a;
```

## decltype(c++11)
```C++
//希望从表达式的类型推断出要定义的变量的类型，但又不想用该表达式的值初始化变量。
//decltype((var))的结果永远是引用，而decltype(var)结果只有当var本身就是一个引用时才是引用。
decltype((var)) a = b;
decltype(var) c;
```

## 引用
引用的目的在于描述函数的参数和返回值，特别是为了运算符的重载。   
普通变量的引用：
```C++
int i = 1;
int& a = i; // 正确
int& b = 1; // 错误
const int& b = 1; // 正确
```
函数的形参使用引用后，函数体就能改变该形参的值。
```C++
void increment(int& i) {
    i++;
}
```
函数返回值的引用
```C++
int& A() {...}
int B() {...}
const int& a = A(); // 编译器不会给返回值临时分配内存
const int& b = B(); // 编译器给返回值临时分配一块内存
```
千万不要返回局部对象的引用，会使引用指向不确定的地址
```C++
const string &A(const string &s) {  
	string ret=s; // ret是局部变量，函数执行完，其地址也释放
	return ret; // 错误，返回它的引用会产生类似野指针的效果
}
```
返回形参的引用
```C++
const string &B(const string &s) {  
	return s;  
}  
```

## string的一些注意点
加号拼接string，要注意不能连续两个字面值常量出现在开头
```C++
string a = "123" + "456"; // 非法
string a = "123" + "456" + b; // 非法
string a = "123" + "456" + b + "abc"; // 非法
string a = "123" + b + "456" + "789"; // 合法
string a = b + "123" + "456" + "789"; // 合法
```

## vector
```C++
vector<T> a;
a.push_back(xx); // 插入元素
a.empty(); // 判断是否为空
a.size(); // 大小
a[i]; // 获取第i个元素
// 遍历的方法1：使用迭代器
for (auto it = a.begin(); it != a.end(); ++it)
    cout << *it << endl;
// 遍历的方法1：不使用迭代器
for (auto i : a)
    cout << i << endl;
```

## 指针数组、数组引用、数组指针
```C++
int *p[10]; // []的优先级高于*，从右向左解读，首先p是一个数组，里面放着10个int型指针
int *(&a)[10] = ptrs; // a是个数组的引用，该数组含有10个int型指针
int (*p)[10]; // ()优先级大于[]，首先p是一个指针，指向一个数组，这个数组中含有10个int型整数
```

## 数组
```C++
int a[] = {0,1,2,3,4};
// 使用迭代器遍历数组
for (auto it = begin(a); it != end(a); ++it)
    cout << *it << endl;
cout << a[-2] << endl; // 注意，数组下标小于0都表示第一个元素，同a[0]
```

## 64位系统下的典型sizeof输出
```text
sizeof(char):   1
sizeof(int):    4
sizeof(float):  4
sizeof(char*):  8
sizeof(int*):   8
sizeof(float*): 8
sizeof(void*):  8
```

## string、const char*、char*、char[]之间的转换
| name  | age | gender    | money  |
|-------|:---:|-----------|-------:|
| rhio  | 384 | robot     | $3,000 |
| haroo | .3  | bird      | $430   |
| jedi  | ?   | undefined | $0     |

## void*自增自减的移动单位
void*自增或者自减，移动单位为内存的最小存储单元（字节）。   
测试程序：
```C++
#include <iostream>
using namespace std;

int main()
{
    int a = 1;
    void *p = &a;
    cout << "before: " << p << endl;
    p++;
    cout << "after: " << p << endl;
    return 0;
}
```
输出结果
```text
before: 0x7ffd31536d4c
after: 0x7ffd31536d4d
```

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
模板还支持模板参数和基本数据类型的混搭：
```C++
template<typename T, int MAXSIZE>
class Stack{
Private:
    T elems[MAXSIZE];
};
```