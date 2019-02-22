/*
* 左值：能够取地址的表达式、变量
* 右值：不能进行取地址操作的任何表达式、变量、常量
* 举例：++i(左值)，i++(右值)，*p(左值)，&a(右值)
*
* 右值引用注意点：
* 1. lambda表达式作为右值引用时，需要像函数调用一样使用，例如a()，a(xxx)
* 2. 尽量在右值引用的情况下，使用auto，避免不必要的报错
* 3. 引用折叠，只能用于函数模板，不能用于普通函数
*/

#include <iostream>
#include <string.h>

template <typename T>
T f1(T a)
{
    return a;
}

template <typename T>
T f2(T &&a)
{
    return a;
}

class A
{
  public:
    A(std::string name, int size) : n(name), _size(size)
    {
        std::cout << n << ", normal constructor" << std::endl;
        data = new int[_size]{1, 2, 3, 4};
    }
    A(const A &a) : n(a.n), _size(a._size), data(a.data)
    {
        std::cout << n << ", copy constructor" << std::endl;
        if (a.data == nullptr)
        {
            data = new int[_size];
            memcpy(data, a.data, _size * sizeof(int));
        }
    }
    A(A &&a) : n(a.n), _size(a._size), data(a.data)
    {
        std::cout << n << ", move constructor" << std::endl;
        data = a.data;
        a.data = nullptr;
    }
    A &operator=(const A &a)
    {
        std::cout << n << ", assign" << std::endl;
        if (a.data == nullptr)
        {
            data = new int[_size];
            memcpy(data, a.data, _size * sizeof(int));
        }
        return *this;
    }
    ~A()
    {
        std::cout << n << ", destructor" << std::endl;
        if (data != nullptr)
            delete[] data;
    }

  private:
    std::string n;
    int _size;
    int *data{nullptr};
};

A getA(std::string name, int size)
{
    A a(name, size);
    return a;
}

void func(int &x) { std::cout << "lvalue ref" << std::endl; }
void func(int &&x) { std::cout << "rvalue ref" << std::endl; }
void func(const int &x) { std::cout << "const lvalue ref" << std::endl; }
void func(const int &&x) { std::cout << "const rvalue ref" << std::endl; }

template <typename T>
void perfectForward(T &&t) { func(std::forward<T>(t)); }

int main()
{
    // 纯右值(prvalue)：字面值、表达式
    // 将亡值(xrvalue)：转换为右值引用的表达式
    int tmp = 0;                 // 普通赋值给变量
    int &&a = 0;                 // 字面量的右值引用，此处的a就是将亡值，不存在任何拷贝操作
    int &&b = f1(1);             // 普通函数的右值引用
    auto &&c = [] { return 2; }; // lambda表达式的右值引用，注意不能写成int &&c = []{return 2;};
    // 引用折叠，模板函数形参为右值引用，可以同时支持传入左值和右值，自动转换
    int &&d = f2(0);
    auto &&e = f2(tmp);

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c() << std::endl; // c实际是一个lambda表达式，需要像函数一样调用
    std::cout << d << std::endl;
    std::cout << e << std::endl;

    /* 转移语义 */
    A a0("a0", 10);
    A a1("a1", 10);                  // a1, normal constructor
    A a2 = a1;                       // a1, copy constructor
    A a3 = std::move(a0);            // move返回右值引用，a0已经正式转移给了a3
    A &&a4 = std::move(a1);          // a4仅仅是a1的右值引用，move并没有发生
    A a5(std::move(a1));             // a1, move constructor
    A a6(A("a6", 10));               // a6, normal constructor
    A a7(getA("a7", 10));            // a7, normal constructor
    A a8(std::move(A("a8", 10)));    // a8, normal constructor   a8, move constructor   a8 destructor
    A a9(std::move(getA("a9", 10))); // a9, normal constructor   a9, move constructor   a9 destructor

    /* 完美转发 */
    perfectForward(10); // rvalue ref
    int x = 1;
    perfectForward(x);            // lvalue ref
    perfectForward(std::move(x)); // rvalue ref

    const int y = 2;
    perfectForward(y);            // const lvalue ref
    perfectForward(std::move(y)); // const rvalue ref

    return 0;
}