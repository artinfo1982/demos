# C语言的若干技巧和经验

## 基本数据类型占用的字节数
```text
Linux32位：long(4)、float(4)、double(8)、long long(8)、long double(8)
Linux64位：long(8)、float(4)、double(8)、long long(8)、long double(16)

long：win32(4)、win64(4)、linux32(4)、linux64(8)
char*：win32(4)、win64(8)、linux32(4)、linux64(8)
size_t：win32(4)、win64(8)、linux32(4)、linux64(8)
```
注意：unsigned float和unsigned double非法。

## const char* 和 char* const
```text
const char* p;
<---------------从右往左看
p是一个指针，指向一个const char（指针指向的值不可改）
char* const p;
<---------------从右往左看
p是一个const指针，指向一个char（指针不可变，但指向的内容可变）
```

## 编译器词法的正负抵消
```C
int a = +-3; // 合法，正负抵消
int b = -+-3; // 合法，正负抵消
int c = ---n; // 非法，优先--，但前面有不能抵消的-
int d = +++n; // 非法，优先++，但前面有不能抵消的+
```

## 有符号和无符号
```C
int a = -1;
printf("%d\n", (unsigned int)a); // -1
printf("%u\n", a); // 4294967295
printf("%u\n", (unsigned int)a); // 4294967295
printf("%d\n", sizeof(a)); // 4
// sizeof返回unsigned int，有符号和无符号混合，有符号转为无符号

unsigned int a = 1;
int b = -2;
if (a+b > 0)
	printf("a+b > 0\n"); // 此为结果
else
	printf("a+b <= 0\n");

unsigned char a = 1;
char b = -2;
if (a+b > 0)
	printf("a+b > 0\n");
else
	printf("a+b <= 0\n"); // 此为结果，char --> int
```

## printf
```text
%d：有符号十进制
%u：无符号十进制
%x：以0x表示十六进制无符号，例如0xabcd
%X：以0X表示十六进制无符号，例如0XABCD
%p：用内存地址的格式打印值
```

## 内定调试宏
```text
__LINE__	当前代码行数
__FILE__	当前文件名，注意安全红线
__DATE__	当前日期
__TIME__	当前时间

例：
#define WARN(str) printf("%s, line=%d, file=%s, date=%s, time=%s", str, __LINE__, __FILE__, __DATE__, __TIME__)
```

## 运算符优先级
```text
+、-、*、/优先级大于 &、|、!
```

## 宏
```text
宏函数的入参，不要是普通函数表达式，可能会多次计算
#define func (x) (x+1)
宏函数禁止函数名和入参间有空格
```

## 枚举
```C
// 不要使用枚举大小（和编译器有关）
// 枚举值不要大于32位

typedef enum
{
	e1 = 1,
	e2
}E1;

typedef enum
{
	e3 = 123456789,
	e4
}E2;

sizeof(E1); // Visual Studio: 4; GCC: 4
sizeof(E2); // Visual Studio: 4; GCC: 8

typedef enum
{
	a1 = 4,
	a2 = 2,
	a3, // 3
	a4 // 4
}A1;

typedef enum
{
	a5 = 4,
	a6, // 5
	a7 = 5,
	a8 = 5
}A2;

// 枚举值可以相同
```

## 结构体和联合体
```C
typedef struct
{
	int a;
	int b;
	int c;
}S;

S s = {1, 2};
printf("%d %d %d\n", s.a, s.b, s.c); // 1, 2, 0

// 默认4字节对齐，可以用#pragma pack(x)指定对齐的字节数

#pragma pack(1)
typedef struct
{
	char a;
	int b;
}S1;

#pragma pack(4)
typedef struct
{
	char a;
	int b;
}S2;
printf("%lu %lu\n", sizeof(S1), sizeof(S2)); / 5, 8

typedef struct
{
	char a;
	short b;
	short c;
}S3;

#pragma pack(4)
typedef struct
{
	char a;
	short b;
	short c;
}S4;
printf("%lu %lu\n", sizeof(S3), sizeof(S4)); // 6, 6

// 结构体自身对齐值和指定对齐值中小的那一个。

typedef struct
{

}S5;
sizeof(S5); // 1

// 单独int a[]; 非法，但却可以在struct中这样写。

typedef struct
{
	int a;
	int b[0];
}S6;

typedef struct
{
	int a;
	int b[];
}S7;
printf("%lu %lu\n", sizeof(S6), sizeof(S7)); // 4, 4

// union的sizeof为里面最大长度类型的长度
```

## 生成时间格式字符串，分别精确到秒和毫秒
```C
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>

/*
生成时间格式字符串，精确到秒，例2018-01-01 10:10:10
*/
char* genTimePrecisionSecond(char* des)
{
	if (NULL == des)
		return NULL;
	time_t t = time(0);
	strftime(des, 20, "%Y-%m-%d %H:%M:%S", localtime(&t));
	return des;
}

/*
生成时间格式字符串，精确到豪秒，例2018-01-01 10:10:10.123
*/
char* genTimePrecisionMillisecond(char* des)
{
	if (NULL == des)
		return NULL;
	struct timeval _tv;
	struct tm _tm;
	gettimeofday(&_tv, NULL);
	localtime_r(&(_tv.tv_sec), &_tm);
	sprintf(des, "%04d-%02d-%02d %02d:%02d:%02d.%03ld", 
		_tm.tm_year+1900, _tm.tm_mon+1, _tm.tm_mday, 
		_tm.tm_hour, _tm.tm_min, _tm.tm_sec, _tv.tv_usec/1000);
	return des;
}


int main(int argc, char* argv[])
{
	char* p1 = (char*)malloc(32*sizeof(char));
	memset(p1, 0x0, 32);
	char* p2 = (char*)malloc(32*sizeof(char));
	memset(p2, 0x0, 32);
	genTimePrecisionSecond(p1);
	genTimePrecisionMillisecond(p2);
	printf("%s\n", p1);
	printf("%s\n", p2);
	free(p1);
	p1 = NULL;
	free(p2);
	p2 = NULL;
	return 0;
}
```

## goto
goto用于跳转到某个特定的区域执行某些特定的代码，比如下面的例子，其中out称为标记：
```C
#include <stdio.h>

void A(int a, int b) {
        if (a < b)
                goto out;
        printf("hello\n");
out:
        printf("use goto\n");
}

int main() {
        printf("---A(1, 2)---\n");
        A(1, 2);
        printf("---A(2, 1)---\n");
        A(2, 1);
        return 0;
}
```
输出为：
```text
---A(1, 2)---
use goto
---A(2, 1)---
hello
use goto
```
可以标记并不能阻止其中的程序段被执行，如果想要达到不需要goto也不想执行goto的程序段时，可以将goto的程序段写在函数的最后，同时标记上方使用return。
```C
#include <stdio.h>

void A(int a, int b) {
        if (a < b)
                goto out;
        printf("hello\n");
        return;
out:
        printf("use goto\n");
}

int main() {
        printf("---A(1, 2)---\n");
        A(1, 2);
        printf("---A(2, 1)---\n");
        A(2, 1);
        return 0;
}
```
输出为：
```text
---A(1, 2)---
use goto
---A(2, 1)---
hello
```

## struct的成员前面使用点（.）
```C
struct point {
        int x, y;
};
struct point p = {
        .y = yvalue,
        .x = xvalue
};
```
这是GCC扩展，用于结构体成员的初始化。详见：
http://gcc.gnu.org/onlinedocs/gcc/Designated-Inits.html   
结构体数组也可以使用类似的方法：
```C
struct point ptarray[10] = {
        [2].y = yv2, 
        [2].x = xv2, 
        [0].x = xv0
};
```
