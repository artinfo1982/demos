/*
一个N*N的二维矩阵，骑士可以在任意的位置作为初始位置，
然后类似中国象棋的马，只可以走日，不断跑直至覆盖棋盘上所有的点
*/

#include <stdio.h>

int pos[8][8] = {0};

int travel(int, int);

int main()
{
    int i, j, startX, startY;
    printf("Please input a starting point: ");
    scanf("%d%d", &startX, &startY);
    if(travel(startX, startY)) {
        printf("Travel finished\n");
    }else {
        printf("Travel failed\n");
    }
    for(i=0; i<8; i++) {
        for(j=0; j<8; j++) {
            printf("%2d ", pos[i][j]);
        }
        printf("\n");
    }
    return 0;
}

int travel(int x, int y) {
    int i, j, k, l, m;
    int tmpX, tmpY;
    int count, min, tmp;

    //骑士可走的八个方向(顺时针)
    int ktmoveX[8] = {1, 2, 2, 1, -1, -2, -2, -1};
    int ktmoveY[8] = {-2, -1, 1, 2, 2, 1, -1, -2};

    //下一步坐标
    int nextX[8] = {0};
    int nextY[8] = {0};

    //记录每个方向的出路的个数
    int exists[8] = {0};

    //起始用1标记位置
    i = x;
    j = y;
    pos[i][j] = 1;

    //遍历棋盘
    for(m=2; m<=64; m++) {
        //初始化八个方向出口个数
        for(l=0; l<8; l++) {
            exists[l] = 0;
        }
        l = 0; //计算可走方向

        //试探八个方向
        for(k=0; k<8; k++) {
            tmpX = i + ktmoveX[k];
            tmpY = j + ktmoveY[k];
            //边界 跳过
            if(tmpX<0 || tmpY<0 || tmpX>7 || tmpY>7) {
                continue;
            }
            //可走 记录
            if(pos[tmpX][tmpY] == 0) {
                nextX[l] = tmpX;
                nextY[l] = tmpY;
                l++;    //可走方向加1
            }
        }
        count = l;
        //无路可走 返回
        if(count == 0) {
            return 0;
        //一个方向可走 标记
        }else if(count == 1) {
            min = 0;
        //找出下个位置出路个数
        }else {
            for(l=0; l<count; l++) {
                for(k=0; k<8; k++) {
                    tmpX = nextX[l] + ktmoveX[k];
                    tmpY = nextY[l] + ktmoveY[k];
                    if(tmpX<0 || tmpY<0 || tmpX>7 || tmpY>7) {
                        continue;
                    }
                    if(pos[tmpX][tmpY] == 0) {
                        exists[l]++;
                    }
                }
            }
            //找出下个位置出路最少的方向
            min = 0;
            tmp = exists[0];
            for(l=0; l<count; l++) {
                if(exists[l] < tmp) {
                    tmp = exists[l];
                    min = l;
                }
            }
        }
        //用序号标记走过的位置
        i = nextX[min];
        j = nextY[min];
        pos[i][j] = m;
    }
    return 1;
}