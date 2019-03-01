1. 常规数学运算
```python
a = x // y  # 返回商的整数部分
a = x ** 3  # x的立方，也可以用pow(x, 3)
c = complex(1, 2)   # 复数，1 + 2j
c.conjugate()   # 求复数的共轭
x += 1  # 相当于x++，python里不支持++、--
bin(x)  # 当且仅当x是整数时，返回其二进制格式

import math
math.ceil(-45.17)   # -45，返回比x大的最小整数
math.ceil(45.17)    # 46，返回比x大的最小整数
math.fabs(-2)   # 2，取x的绝对值
math.factorial(x)   # 返回x的阶乘
math.floor(-45.17)  # -46，返回比x小的最大整数
math.floor(45.17)   # 45，返回比x小的最大整数
math.fmod(x, y) # 返回x对y取模的余数
math.fsum(x)    # 返回一维向量x的各项和，注意仅能在一维使用。例如[2, 3, 4]
math.isinf(x)   # 如果x是正负无穷大，返回True
math.isnan(x)   # 如果x是NaN，返回True
math.modf(x)    # 返回x的小数部分和整数部分，元组的形式，例如modf(3.14)，返回(14, 3)
math.exp(x) # 返回e**x
math.log(x, a)  # 返回a为底x的对数，不指定a，则默认以e为底
math.log2(x)    # 返回2为底x的对数
math.log10(x)   # 返回10为底x的对数
math.sqrt(x)    # 返回x的平方根
math.trunc(x)   # 直接取x的整数部分
```
2. 逻辑操作
```python
a and b
a or b
if not a
if a == 1
if a != 1
```
3. 字符串操作
```python
s.replace('aaa', 'bbb') # 把字符串s中所有的aaa替换为bbb
s.strip()   # 做trim
s.startswith('aaa') # 如果字符串以aaa开头，返回True
s.endswith('aaa')   # 如果字符串以aaa结尾，返回True
str1.find(str2) # 返回str1中首次出现str2的位置，如果没有，返回-1
'-'.join(("a", "b", "c"))   # 以'-'为分隔符，将序列中的所有元素拼接在一起，形成一个新的字符串，a-b-c
s *= n  # 将字符串扩展n次，拼接
s.count(x)  # 在字符串s中统计x出现的次数
```
4. 容器
```python
# List，数组，长度可变，可以存放不同类型的元素
```