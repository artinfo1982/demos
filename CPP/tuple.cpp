/*
* 演示std::tuple的一般用法。
* tuple，元组，通常用于函数返回多个不同类型的参数。
*/

#include <iostream>
#include <tuple>
#include <typeinfo>

std::tuple<int, float, std::string> func(void)
{
    return std::make_tuple<int, float, std::string>(1, 2.2, "aaa");
}

/* 打印类型的通用函数 */
template <typename T>
void printType(T &t)
{
    if (typeid(t) == typeid(int))
        std::cout << "int";
    else if (typeid(t) == typeid(float))
        std::cout << "float";
    else if (typeid(t) == typeid(std::string))
        std::cout << "string";
    else if (typeid(t) == typeid(const char *))
        std::cout << "const char*";
    else
        std::cout << "unkown";
}

/* 遍历tuple，打印每个元素的类型、值 */
template <typename Tuple, std::size_t N>
struct TuplePrinter
{
    static void print(const Tuple &tup)
    {
        TuplePrinter<Tuple, N - 1>::print(tup);
        auto t = std::get<N - 1>(tup);
        std::cout << "type=";
        printType(t);
        std::cout << ", value=" << t << std::endl;
    }
};

template <typename Tuple>
struct TuplePrinter<Tuple, 1>
{
    static void print(const Tuple &tup)
    {
        auto t = std::get<0>(tup);
        std::cout << "type=";
        printType(t);
        std::cout << ", value=" << t << std::endl;
    }
};

template <typename... Args>
void printTuple(const std::tuple<Args...> &t)
{
    TuplePrinter<decltype(t), sizeof...(Args)>::print(t);
}

int main()
{
    auto tup = func();
    auto size = std::tuple_size<decltype(tup)>::value;
    std::cout << "tuple size=" << size << std : endl;
    printTuple(tup);
    return 0;
}