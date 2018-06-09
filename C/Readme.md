# C语言的若干技巧和经验

## goto
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
