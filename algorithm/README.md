# ASCII表
```text
48-57：0-9
65-90：A-Z
97-122：a-z
大写字母 + 32 = 对应的小写字母
```

# 判定一个数是不是2的整数幂
```C
bool check()
{
	return n & n-1 == 0 ? true : false;
}
```
