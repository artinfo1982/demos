/*
假设有一条绳子,上面有红、白、蓝三种颜色的旗子,起初绳子上的旗子颜色并没有顺序,您
希望将之分类,并排列为蓝、白、红的顺序,要如何移动次数才会最少,注意您只能在绳子上
进行这个动作,而且一次只能调换两个旗子。
*/

#include <stdio.h>

void swap(char array[], int x, int y)
{
    array[x] = array[x] ^ array[y];
    array[y] = array[x] ^ array[y];
    array[x] = array[x] ^ array[y];
}

int main()
{
    char array[10] = {'w', 'b', 'r', 'r', 'w', 'w', 'r', 'b', 'b', 'r'};
    int i, j;
    for (i = 0; i < 10; ++i)
    {
        for (j = i; j < 10; ++j)
        {
            if ((array[i] == 'w' && array[j] == 'b') || (array[i] == 'r' && array[j] == 'w') || (array[i] == 'r' && array[j] == 'b'))
                swap(array, i, j);
        }
    }

    for (i = 0; i < 10; ++i)
        printf("%c ", array[i]);
    printf("\n");

    return 0;
}