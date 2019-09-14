/*
一个NxN的二维矩阵，分布着随机正整数（豆子数），一只仓鼠从左下角出发，
只能向右或者向上移动，直至右上角结束，请列出仓鼠可以走的所有路线，
标出能吃到最多豆子的路线。

动态规划路径算法，使用一个和原始矩阵大小一样的矩阵，记录走过的路径的运算结果。
按照本题的场景，起始为(3,0)，结束为 (0, 3)。
最大路径记录矩阵初始化：
X 0 0 0
X 0 0 0
X 0 0 0
X X X X
最小路径记录矩阵初始化：
X X X X
0 0 0 X
0 0 0 X
0 0 0 X
*/

#include <stdio.h>

int map[4][4] = {
    {22, 92, 68, 71},
    {77, 18, 93, 18},
    {43, 76, 69, 3},
    {20, 76, 33, 8}
};

int ok1 = 0;
int ok2 = 0;
// dp1: 最大路径，dp2：最小路径
int dp1[4][4] = {0};
int dp2[4][4] = {0};

int max(int x, int y)
{
    return x > y ? x : y;
}

int min(int x, int y)
{
    return x < y ? x : y;
}

// 打印最大路径结果记录矩阵中的具体路线
void move1(int i, int j)
{
    printf("(%d,%d) ", i, j);
    if (i == 0 && j == 3)
    {
        ok1 = 1;
        return;
    }
    if (ok1 != 1)
    {
        if (i > 0 && dp1[i-1][j] >= dp1[i][j+1])
            move1(i-1, j);
        else if (j < 3 && dp1[i-1][j] < dp1[i][j+1])
            move1(i, j+1);
        else if (j == 3 && dp1[i-1][3] > dp1[i][3])
            move1(i-1, 3);
        else if (i == 0 && dp1[0][j+1] > dp1[0][j])
            move1(0, j+1);
    }
}

// 打印最小路径结果记录矩阵中的具体路线
void move2(int i, int j)
{
    printf("(%d,%d) ", i, j);
    if (i == 0 && j == 3)
    {
        ok2 = 1;
        return;
    }
    if (ok2 != 1)
    {
        if (i > 0 && dp2[i-1][j] <= dp2[i][j+1])
            move2(i-1, j);
        else if (j < 3 && dp2[i-1][j] > dp2[i][j+1])
            move2(i, j+1);
        else if (j == 3 && dp2[i-1][3] < dp2[i][3])
            move2(i-1, 3);
        else if (i == 0 && dp2[0][j+1] < dp2[0][j])
            move2(0, j+1);
    }
}

int main()
{
    int i, j;

    // 计算最大路径结果记录矩阵
    dp1[3][0] = map[3][0];
	for (i = 2; i >= 0; --i)
		dp1[i][0] = dp1[i+1][0] + map[i][0];
	for (j = 1; j < 4; ++j)
		dp1[3][j] = dp1[3][j-1] + map[3][j];

    for (i = 2; i >= 0; --i)
    {
		for (j = 1; j < 4; ++j)
            dp1[i][j] = max(dp1[i+1][j], dp1[i][j-1]) + map[i][j];
	}

    // 计算最小路径结果记录矩阵
    dp2[0][3] = map[0][3];
	for (i = 1; i < 4; ++i)
		dp2[i][3] = dp2[i-1][3] + map[i][3];
	for (j = 2; j >= 0; --j)
		dp2[0][j] = dp2[0][j+1] + map[0][j];

    for (i = 1; i < 4; ++i)
    {
		for (j = 2; j >= 0; --j)
            dp2[i][j] = min(dp2[i-1][j], dp2[i][j+1]) + map[i][j];
	}

    // 打印dp
    for (i = 0; i < 4; ++i)
    {
        for (j = 0; j < 4; ++j)
            printf("%d ", dp1[i][j]);
        printf("\n");
    }
    printf("\n");
    for (i = 0; i < 4; ++i)
    {
        for (j = 0; j < 4; ++j)
            printf("%d ", dp2[i][j]);
        printf("\n");
    }
    printf("\n");

    // 打印最大路径及最大值
    move1(3, 0);
    printf("\nmax = %d\n", dp1[0][3]);

    // 打印最小路径及最小值
    move2(3, 0);
    printf("\nmin = %d\n", dp2[3][0]);
    return 0;
}