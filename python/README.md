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
4. enumerate，输出索引号和对应的值
```python
for i, j in enumerate([1, 2, 3]):
    print("%d, %d" % (i, j))
```
运行结果
```text
0, 1
1, 2
2, 3
```
5. eval，运行字符串表达式
```python
print(eval('2+2'))
x=1
print(eval('x*3'))
```
6. *args、**kargs
```python
def func1(*args):
    for arg in args:
        print(arg)
def func2(**kwargs):
    for key, value in kwargs.items():
        print('key=%s, value=%s' % (key, value))
# *args
func1('aaa', 'bbb', 'ccc')
# **kwargs用法1
func2(a='1', b='2')
# **kwargs用法2
kwargs = {"a": 1, "b": "2"}
func2(**kwargs)
```
运行结果
```text
aaa
bbb
ccc
key=a, value=1
key=b, value=2
key=a, value=1
key=b, value=2
```
7. 容器
```python
# List，数组，长度可变，可以存放不同类型的元素
list1 = []
list1.append('aaa')
list1.append('bbb')
list1.append('ccc')
'''
# 遍历
for l in list1:
    print(l)
print(list1[2]) # 索引为2
print(list1[-2]) # 倒数第二个
print(list1[1:]) # 从索引1开始到最后
cmp(list1, list2) # 比较两个列表的元素
len(list) # 列表元素个数
max(list) # 返回列表元素最大值
min(list) # 返回列表元素最小值
list.append(obj) # 在列表末尾添加新的对象
list.count(obj) # 统计某个元素在列表中出现的次数
list.extend(seq) # 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
list.index(obj) # 从列表中找出某个值第一个匹配项的索引位置
list.insert(index, obj) # 将对象插入列表
list.pop([index=-1]) # 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
list.remove(obj) # 移除列表中某个值的第一个匹配项
list.reverse() # 反向列表中元素
list.sort(cmp=None, key=None, reverse=False) # 对原列表进行排序
'''

# tuple，元组，数据不可更改
tup = (1, 2, 3)
'''
print(tup[2])
print(tup[-2])
print(tup[1:])
cmp(tuple1, tuple2) # 比较两个元组元素
len(tuple) # 计算元组元素个数
max(tuple) # 返回元组中元素最大值
min(tuple) # 返回元组中元素最小值
'''
```