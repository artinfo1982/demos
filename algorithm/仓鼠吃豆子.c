/*
一个NxN的二维矩阵，分布着随机正整数（豆子数），一只仓鼠从左下角出发，
只能向右或者向上移动，直至右上角结束，请列出仓鼠可以走的所有路线，
标出能吃到最多豆子的路线。
*/

#include <stdio.h>

#define MIN(x, y) (x) < (y) ? (x) : (y)
#define MAX(x, y) (x) > (y) ? (x) : (y)

static int map[4][4] = {
    {22, 92, 68, 71},
    {77, 18, 93, 18},
    {43, 76, 69, 3},
    {20, 76, 33, 8}
};

static int start_x = 3, start_y = 0;
static int end_x = 0, end_y = 3;
static int success = 0;
static int sum = 0;

static void move(int i, int j)
{
    printf("(%d,%d)[%d] ", i, j, map[i][j]);
    sum += map[i][j];
    if (i == end_x && j == end_y)
        success = 1;
    if (success != 1)
    {
        if (map[i-1][j] > map[i][j+1])
            move(i-1, j);
        else if (map[i-1][j] < map[i][j+1])
            move(i, j+1);
        else
        {
            if (MIN(map[i-2][j], map[i-1][j+1]) >= MAX(map[i-1][j+1], map[i][j+2]))
                move(i-1, j);
            else
                move(i, j+1);
        }
    }
}

int main()
{
    move(start_x, start_y);
    printf("\n");
    printf("sum = %d\n", sum);
    return 0;
}