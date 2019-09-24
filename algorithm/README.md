# ASCII表
```text
48-57：0-9
65-90：A-Z
97-122：a-z
大写字母 + 32 = 对应的小写字母
```

# 判定一个数是不是2的整数幂
```C
bool check(long n)
{
	return n & n-1 == 0 ? true : false;
}
```

# 求两个数的最大公约数、最小公倍数
```C
// 最大公约数
unsigned long long gcd(unsigned long long a, unsigned long long b)
{
	return b == 0 ? a : gcd(b, a%b);
}
// 最小公倍数 = 两数乘积 / 最大公约数
unsigned long long gys(unsigned long long a, unsigned long long b)
{
	return a*b/gcd(a, b);
}
```
