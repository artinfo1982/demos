/*
* lambda表达式，也称为匿名函数
*/

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

std::function<int(std::vector<int>)> sum = [](std::vector<int> const &vec) -> int {
    int total = 0;
    std::for_each(vec.begin(), vec.end(), [&total](int v) { return total += v; });
    return total;
};

auto bd1 = std::bind(sum, std::placeholders::_1); // placeholders占位符，从_1开始递增

std::function<int(int, int)> min = [](const int &x, const int &y) -> int { return x < y ? x : y; };
auto bd2 = std::bind(min, 3, std::placeholders::_1);

int main()
{
    int total = 0;
    auto a = [](int x, int y) -> int { return x + y; }; // 标准格式 [捕获值列表]{形参列表}->返回值类型{函数体}
    std::cout << a(1, 2) << std::endl;
    auto b = [](int x, int y) { return x + y; }; // 简化格式，省掉->返回值类型
    std::cout << b(1, 2) << std::endl;
    auto c = [](int x) { std::cout << x << std::endl; }; // 如果没有return，则函数的返回值类型为void
    c(2);
    std::vector<int> num;
    for (int i = 1; i <= 5; ++i)
        num.push_back(i);
    std::for_each(num.begin(), num.end(), [&total](int v) { total += v; }); // 带捕获值的应用，[捕获值]，捕获值可以理解为对外部变量的操作
    std::cout << total << std::endl;

    std::cout << sum(num) << std::endl;
    std::cout << bd1(num) << std::endl;
    std::cout << min(10, 8) << std::endl;
    std::cout << bd2(2) << std::endl;

    return 0;
}