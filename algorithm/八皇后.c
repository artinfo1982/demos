/*
位运算求解八皇后
*/
#include<stdio.h>
#include<time.h>
 
void queen(long ld, long rd);
 
//row的位表示每一列的状态，为1表示该列放有了皇后。sum表示摆放n皇后的不同摆法。
//uplimit用来记录
long row, sum, uplimit, upperlim;	
										
int n; //表示n皇后
void main()
{
	long ld, rd;	//ld中位为1的表示反斜杠线上不能放皇后的位置，rd的1表示斜杠线上不能放皇后的位置
	row = sum = ld = rd = 0;
	clock_t start_t, finish_t;   //用于计算程序运行的时间
	double user_t;
 
	printf("输入：");
	scanf("%d", &n);
	uplimit = upperlim = (1 << n) - 1;
 
	start_t = clock();
	queen(ld, rd);
	finish_t = clock();
 
	printf("求%d皇后的所需时间：%f s\n", n, (finish_t - start_t) / (double)CLOCKS_PER_SEC);
	printf("%ld\n", sum);
}
 
void queen(long ld, long rd)
{
	long pos = ~(row | ld | rd);	//算出当前行上那些位置已摆放有皇后了
    //rd向右移位时最高位会填补1，所以为了保证rd左边的sizeof(long)*8-n位全为0，必须要进行这步运算
    //因为ld向左移位是最低位填补0，ld不需要进行这步运算
	rd = rd&upperlim;	
	pos = pos&upperlim;	//由于ld左边的sizeof(long)*8-n有些为1，所以为了保证rd左边的sizeof(long)*8-n位全为0，必须要进行这步运算
	if (pos != 0)	//判断该行上是不是全部都摆放了皇后
	{
		while (pos != 0)
		{
			long p = pos&(~pos + 1);	//从当前行可以摆放皇后的位置中选择最右的一位
			p = p&upperlim;
			pos -= p;	//去除上上步操作里选择过的那位
			row = row | p;
			if (uplimit != row)		//判断该行是不是最后一行，若是sum++，不是则进入下一行
			{
				queen((ld | p) << 1, (rd | p) >> 1);
			}
			else
				sum++;
			row = row & ~p;
		}
	}
}
