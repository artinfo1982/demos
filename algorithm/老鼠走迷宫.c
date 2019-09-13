/*
我们在二维阵列中使用2表示迷宫墙壁,使用1来表示老鼠的行走路径,试以程式求出由入口至出口的路径。
老鼠的走法有上、左、下、右四个方向,在每前进一格之后就选一个方向前进,
无法前进时退回选择下一个可前进方向,如此在阵列中依序测试四个方向,直到走到出口为止
*/

#include <stdio.h>
#include <stdlib.h>

int maze[7][7] = {
    {2, 2, 2, 2, 2, 2, 2},
    {2, 0, 0, 0, 0, 0, 2},
    {2, 0, 2, 0, 2, 0, 2},
    {2, 0, 0, 2, 0, 2, 2},
    {2, 2, 0, 2, 0, 2, 2},
    {2, 0, 0, 0, 0, 0, 2},
    {2, 2, 2, 2, 2, 2, 2}
};
int startI = 1, startJ = 1; // 入口
int endI = 5, endJ = 5; // 出口
int success = 0;

int move(int i, int j)
{
    maze[i][j] = 1;
    if(i == endI && j == endJ)
        success = 1;
    if(success != 1 && maze[i][j+1] == 0)
        move(i, j+1);
    if(success != 1 && maze[i+1][j] == 0)
        move(i+1, j);
    if(success != 1 && maze[i][j-1] == 0)
        move(i, j-1);
    if(success != 1 && maze[i-1][j] == 0)
        move(i-1, j);
    if(success != 1)
        maze[i][j] = 0;
    return success;
}

int main()
{
    int i, j;
    if(move(startI, startJ) == 0)
        printf("\n没有找到出口!\n");
    else
    {
        for(i = 0; i < 7; i++)
        {
            for(j = 0; j < 7; j++)
            {
                if(maze[i][j] == 2)
                    printf("#");
                else if(maze[i][j] == 1)
                    printf("*");
                else
                    printf(" ");
            }
            printf("\n");
        }
    }
    return 0;
}