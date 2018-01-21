#include <stdio.h>
#include <limits.h>

/*
* 求任意一个整数数列中的最大值、最小值、平均值
*/

// 比大小的宏
#define MAX(x, y) ({\
        typeof(x) _x = x;\
        typeof(y) _y = y;\
        (void)(&_x == &_y);\
        _x>_y?_x:_y;\
})

#define MIN(x, y) ({\
        typeof(x) _x = x;\
        typeof(y) _y = y;\
        (void)(&_x == &_y);\
        _x<_y?_x:_y;\
})

int main(int argc, char *argv[])
{
        int i;
        int max = INT_MIN, min = INT_MAX;
        int sum = 0;
        double avg = 0.00;
        int n[argc-1];
        for (i=1; i<argc; i++)
        {
                n[i-1] = atoi(argv[i]);
        }
        for (i=0; i<argc-1; i++)
        {
                max = MAX(max, n[i]);
                min = MIN(min, n[i]);
                sum += n[i];
        }
        avg = (double)sum/(argc-1);
        printf("max=%d, min=%d, avg=%.2f\n", max, min, avg);
        return 0;
}
