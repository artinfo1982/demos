/*
* 演示std::bitset的一般用法。
* bitset，c++位操作
*/

#include <iostream>
#include <bitset>

int main()
{
    std::bitset<10> b1;
    b1 = 8;
    std::cout << b1 << std::endl; // 二进制形式输出
    // bool any()，只要有至少一位是1，返回true
    if (b1.any())
        std::cout << "has 1" << std::endl;
    // bool all()，所有的位都是1，返回true
    if (b1.all())
        std::cout << "all 1" << std::endl;
    // size_type count()，返回位是1的个数
    std::cout << b1.count() << std::endl;
    // bool none()，如果都是0，返回true
    if (b1.none())
        std::cout << "all 0" << std::endl;
    // string to_string()，以字符串形式输出bitset
    std::cout << b1.to_string() << std::endl;
    // unsigned long to_ulong()，返回bitset的无符号长整数形式
    std::cout << b1.to_ulong() << std::endl;
    // bool test(size_t pos)，pos位置是1，返回true
    if (b1.test(3))
        std::cout << "pos 3 is 1" << std::endl;
    b1.flip(); // 所有位翻转，1->0，0->1，flip(x)也可以指定位翻转
    std::cout << b1.to_string() << std::endl;

    std::bitset<10> b2("1100"); // 也可以直接从string来初始化
    b2.set(1, 1);               // 将某一位设置为1
    b2.set();                   // 将所有位设置为1
    b2.reset(0);                // 将右边第一位设置为0
    b2.reset();                 // 重置所有位为0

    return 0;
}