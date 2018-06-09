# C语言的若干技巧和经验

## goto
goto用于跳转到某个特定的区域执行某些特定的代码，比如下面的例子，其中out称为标记：
```C
#include <stdio.h>

void A(int a, int b)
{
        if (a < b)
                goto out;
        printf("hello\n");
out:
        printf("use goto\n");
}

int main()
{
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

void A(int a, int b)
{
        if (a < b)
                goto out;
        printf("hello\n");
        return;
out:
        printf("use goto\n");
}

int main()
{
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
struct point
{
        int x, y;
};
struct point p = 
{
        .y = yvalue,
        .x = xvalue
};
```
这是GCC扩展，详见：
http://gcc.gnu.org/onlinedocs/gcc/Designated-Inits.html
