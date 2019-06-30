# C++的若干经验技巧
1. 尽量使用++i，不要用i++（除非必须要用），因为i++会将i的值先存起来
2. 尽量代码简洁
```text
*p++; // 先取*p，再p++
```
3. <<、>>的优先级大于<、>，而小于+、-、*、/
4. 形参和实参。int func(int a); func(3); a是形参，3是实参
5. 函数内的局部静态变量生命期为main函数销毁时，而非本函数结束
6. 函数调用分传值、传引用。传指针也是传值的一种，因为指针里面存放着对象的地址。传值不会改变实参的值（通过指针修改的除外），是复制实参给形参。传指针也可以成为传址。尽量避免调用时拷贝对象，用传引用或者传址
7. 不能将const实参传给非const形参，但非const实参可以传给const形参
```C++
// 代码段1
void func1(const int &x) {}
void func2(int &x) {}
int a = 1;
const int b = 2;
func1(a); // 正确
func1(b); // 正确
func2(a); // 正确
func2(b); // 错误，禁止const实参传给非const形参

// 代码段2
void func1(const string &x) {}
void func2(string &x) {}
func1("123"); // 正确
func2("123"); // 错误，"123"是const类型，禁止const实参传给非const形参

// 代码段3
void func1(const int x)
{
    ++x; // 错误，const无论是引用还是非引用，都不能修改值
}
void func2(const int &x)
{
    ++x; // 错误，const无论是引用还是非引用，都不能修改值
}
```
8. 不要返回函数局部变量的引用或指针，可以返回形参的引用。函数返回引用为左值，可以直接运算
```C++
int& func(int &x) {return x;}
int main()
{
    int a = 1;
    cout << ++func(a) << endl;
    return 0;
}
```
9. 函数重载（overload），指的是同名函数，但形参列表不同
10. 指针函数（返回指针的函数），函数指针（指向函数的指针，例如 int (*pf)(const string &, const int&);）
11. this指向const对象
12. const成员函数不能修改成员变量的值（除非是mutable类型）
13. const类对象可以调const成员函数，但不可以调非const成员函数；非const类对象没有任何限制
14. c++11允许用default来生成默认构造函数
```C++
public:
    A() = default;
```
15. 优先使用delete关键字删除函数而不是private却又不实现的函数
```text
不要使用：
private:
    func(xxx); // 没有实现
应该使用：
public:
    func(xxx) = delete;
```
16. 使用override关键字声明覆盖的函数
17. 优先使用声明别名（using）而不是typedef
```C++
#include <iostream>
#include <vector>

template <typename T>
using MyVector = std::vector<std::pair<T, std::string>>;

int main()
{
    MyVector<int> a;
    return 0;
}
```
18. 优先使用作用域限制的 enum class 而不是无作用域的 enum
```C++
#include <iostream>

enum class Color
{
    red,
    green,
    blue
};

int main()
{
    auto c = Color::blue;
    switch(c)
    {
    case Color::red:
        std::cout << "red" << std::endl;
        break;
    case Color::green:
        std::cout << "green" << std::endl;
        break;
    case Color::blue:
        std::cout << "blue" << std::endl;
        break;
    default:
        std::cout << "other" << std::endl;
        break;
    }
    return 0;
}
```
19. 优先使用const_iterator而不是iterator，优先使用cbegin、cend，cbegin和cend实现了const_iterator
```C++
#include <iostream>
#include <vector>

int main()
{
    std::vector<int> values;
    auto it = std::find(values.cbegin(), values.cend(), 1983);
    values.insert(it, 1998);
    return 0;
}
```
20. C/C++里几种常见数据类型最大最小值的宏定义
```C++
#include <float.h>
#include <limits.h>

int n1 = INT_MIN;
int n2 = INT_MAX;
float f1 = FLT_MIN;
float f2 = FLT_MAX;
double d1 = DBL_MIN;
double d2 = DBL_MAX;
long ln1 = LONG_MIN;
long ln2 = LONG_MAX;
long long lln1 = LONG_LONG_MIN;
long long lln2 = LONG_LONG_MAX;
```

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
## 打印十六进制、打印精度
```C++
#include <iostream>
#include <iomanip>

int main()
{
    // 打印十六进制数，不足的左补0
    std::cout << "0X" << std::setfill('0') << std::setw(2) << std::hex << 10 << std::endl;

    // 设置输出精度为5个有效数字
    std::cout << std::setprecision(5) << 3.1415926L << std::endl;

    return 0;
}
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
参考如下的帖子：   
https://blog.csdn.net/rongrongyaofeiqi/article/details/52442169
```text
string-->const char*: string.c_str();
string-->char*: 第一步，string.c_str(); 第二步，char*=<const_cast><char*>(const char*);
string-->char[]: for (int i=0; i<string.length(); ++i) char[i]=string[i];
const char*-->string: 直接转换
char*-->string: 直接转换
char[]-->string: 直接转换
const char*-->char*: char*=<const_cast><char*>(const char*);
const char*-->char[]: strncpy(char, const char*, n);
char*-->char[]: strncpy(char, char*, n);
```
## 二维数组遍历
```C++
int a[2][2] = {{0,1},{2,3}};
for (const auto &row : a) // 避免自动将数组转为指针
    for (auto col : row)
        cout << col << endl;
```
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
## 显式转换
static_cast<类型>(表达式)
```text
不安全，因为没有校验，只要没有const变量。
一个重要的作用：还原void*的原始类型。
例：
void *p = &d; // d为double型
double *dp = static_cast<double*>(p);
```
dynamic_cast<类型>(表达式)
```text
和static不一样的是，它在运行态转换，相对安全些。
```
const_cast<类型>(表达式)
```text
作用：去掉const。
例：
char *p1 = const_cast<char*>(p); // p为const char*类型
```
reinterpret_cast<类型>(表达式)
```text
尽量少用，采用位模式重新诠释。
例：
int *ip;
char *pc = reinterpret_cast<char*>(ip); // 采用char*来表达int*，不安全
```
## try-catch
```C++
// C++安全编程规范要求尽量不用try-catch，需要程序员自己对程序的细节予以全面掌控
try {
    xxx
} catch(exception e) {
    cout << e.what() << endl;
}
```
## 随机
```C++
#include <iostream>
#include <random>
#include <ctime>

int main()
{
    // 生成简单的，10个unsigned的随机整数
    std::default_random_engine e(time(0));
    for (int i = 0; i < 10; ++i)
        std::cout << e() << " ";
    std::cout << std::endl;

    // 生成10个0-100均匀分布的整数
    std::uniform_int_distribution<unsigned> u1(0, 100);
    for (int i = 0; i < 10; ++i)
        std::cout << u1(e) << " ";
    std::cout << std::endl;

    // 生成10个-10到10均匀分布的浮点数
    std::uniform_real_distribution<double> u2(-10, 10);
    for (int i = 0; i < 10; ++i)
        std::cout << u2(e) << " ";
    std::cout << std::endl;

    // 生成10个均值0，方差1的正态分布的浮点数
    std::normal_distribution<double> u3(0, 1.0);
    for (int i = 0; i < 10; ++i)
        std::cout << u3(e) << " ";
    std::cout << std::endl;

    return 0;
}
```
## 处理变长形参列表（initializer_list、可变参数模板、stdarg）
使用c++11的initializer_list
```C++
#include <iostream>
#include <string>
#include <initializer_list>

using namespace std;

void func(initializer_list<string> li)
{
    for (auto it=li.begin(); it!=li.end(); ++it)
    {
        if (*it == "-a")
            cout << "param a value: " << *(++it) << endl;
        else if (*it == "-b")
            cout << "param b value: " << *(++it) << endl;
        else if (*it == "-c")
            cout << "param c value: " << *(++it) << endl;
    }
}

int main()
{
    func({"-a", "1", "-b", "2", "-c", "3"});
    return 0;
}
```
输出
```text
param a value: 1
param b value: 2
param c value: 3
```
使用c++11的可变参数模板
```C++
#include <iostream>
#include <type_traits>

// 递归终止函数
void func() {}

// 展开函数
template <typename T, typename... Args>
void func(T head, Args... args)
{
    if (std::is_same<T, int>::value)
    {
        std::cout << "int type" << std::endl;
    }
    else if (std::is_same<T, float>::value)
    {
        std::cout << "float type" << std::endl;
    }
    else if (std::is_same<T, std::string>::value)
    {
        std::cout << "string type" << std::endl;
    }
    else if (std::is_same<T, const char*>::value)
    {
        std::cout << "const char* type" << std::endl;
    }
    else
    {
        std::cout << "other type" << std::endl;
    }
    func(args...);
}

int main()
{
    func("-a", 1, "-b", 2.2f);
    return 0;
}
```
使用stdarg.h
```C++
#include <iostream>
#include <stdarg.h>

void demo(int num, ...)
{
    std::cout << "input params: " << num << std::endl;
    va_list valist;
    const char* s;
    va_start(valist, num);
    for (int i=0; i<num; ++i)
    {
        s = va_arg(valist, const char*);
        std::cout << s << std::endl;
    }
    va_end(valist);
}

int main()
{
    demo(3, "aaa", "bbb", "ccc");
    return 0;
}
```
## 友元函数和友元类
```C++
class A {
    friend void b() {}; // 友元函数，类内位置不限
    friend void B::func(); // 也可以设置某类的某个成员函数为友元
    public:
        void a() {}; // 成员函数
};
void c() {}; // 非成员函数
class B {
    friend class C; // C是B的友元类，C的所有成员函数都可以访问B的私有变量
    void func() {};
    ...
};
class C {
    ...
};
```
## 构造函数初始化列表
```C++
class A {
public:
    A(int a, int b):x(a), y(b) {}; // 构造函数初始化列表
    ...
private:
    int x;
    int y;
};
```
## 委托构造函数（c++11）
```C++
class A {
public:
    A(int a, int b, int c):x(a), y(b), z(c) {}; // I
    A(int a, int b):A(a, 0, 0) {}; // II
    A(int a):A(a, 0) {}; // III
    ...
};
// II 委托 I，III 委托 II
// 被委托的构造函数应该包含较大数量的参数，初始化较多的成员变量
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
## C和C++混合编程
```C++
#ifdef __cplusplus
extern "C"
{
    XXX
}
#endif
// 上面这个结构实现C/C++混合编程。
// extern "C"的作用是告诉C++程序下面这段代码是C代码，编译时不使用C++名字修饰，而应该使用C的修饰。
```
C调用C++   
a.cpp a.h main.c Makefile   
a.h
```C++
#ifndef __A_H__
#define __A_H__

#ifdef __cplusplus
extern "C"
{
#endif
class A {
    ...
}
void print(void); // 必须有一个类外函数，才能被C调用
#ifdef __cplusplus
}
#endif
#endif
```
a.cpp
```C++
#include "a.h"
...
void print(void)
{
    ...
}
```
main.c
```C
int main()
{
    print();
    return 0;
}
```
Makefile
```text
main: main.c a.o
    gcc -lstdc++ main.c a.o -o main
a.o: a.h
    g++ -c a.cpp
```
C++调用C   
a.h a.c main.cpp   
a.h
```C++
#ifndef __A_H__
#define __A_H__
extern void print(char*);
#endif
```
a.c
```C
#include <stdio.h>
#include "a.h"

void print(char *data)
{
    printf("%s\n", data);
}
```
main.cpp
```C++
extern "C"
{
    #include "a.h"
}
int main()
{
    print("aaa\n");
    return 0;
}
```
```shell
gcc -c a.c
g++ main.cpp a.o
```
## 编译动态链接库
```shell
gcc test.c -fPIC -shared -o libtest.so
g++ test.cpp -fPIC -shared -o libtest.so
```
## Makefile示例
```text
all: A B
A:
    g++ a.cpp a.h -fPIC -shared -o liba.so
B:
    g++ b.cpp b.h -L./lib -la -o bin/test
clean:
    rm bin/test
.PHONY: all
```
```shell
make clean
make all
```