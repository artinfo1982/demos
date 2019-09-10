#include <iostream>
#include <math.h>

bool isPrime(int num)
{
    if (num == 2 || num == 3)
        return true;
    //如果不在6的倍数附近，肯定不是素数
    if (num % 6 != 1 && num % 6 != 5)
        return false;
    //对6倍数附近的数进行判断
    for (int i = 5; i <= sqrt(num); i += 6)
    {
        if (num % i == 0 || num % (i + 2) == 0)
            return false;
    }
    return true;
}

int main()
{
    int n;
    while (scanf("%d", &n))
    {
        if (isPrime(n))
            std::cout << n << std::endl;
    }
    return 0;
}