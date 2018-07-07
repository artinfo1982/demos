# C语言的若干技巧和经验

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
