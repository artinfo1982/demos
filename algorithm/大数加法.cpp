#include <iostream>
#include <string>

using namespace std;

std::string add(std::string x, std::string y)
{
	if(x.size()<y.size()) //把x固定为位数较大的那个数，方便后面处理
    {
		std::string temp = x;
		x = y;
		y = temp;
	}
	int length1 = x.size(), length2 = y.size();
    int flag = 0, a, b, sum; //flag是进位标记
	while(length1 > 0) //从低位开始把对应的位相加
    {
		a = x[length1 - 1] - '0'; //获取x当前位的数字
		if(length2 > 0) //如果y还没加完（注意，y是位数较少的）
			b = y[length2 - 1] - '0'; //获取y当前位的数字
		else
			b = 0; //如果y加完了，y对应位上就没有数来加了
		//这时我没有break，因为虽然y没有数字来加了，但可能还有进位需要加
		sum = a + b + flag; //x与y对应位上的数字相加，再加上进位位
		if(sum >= 10) //如果加起来大于于10，那就需要进位了
        {
			x[length1 - 1] = '0' + sum % 10; //计算加完之后，当前位应该是多少
			flag = 1; //把进位标记置1
		}
        else
        {
			x[length1 - 1] = '0' + sum; //计算加完之后，当前位应该是多少
			flag = 0; //把进位标记置0
		}
		length1--; //向高位移动1位
		length2--; //向高位移动1位
	}
	//如果两个数对应位都加完了，进位位是1，说明位数要增加1了
	//比如99+1，加完之后，变成了三位数100，其实就是再在前面加一位1
	if(1 == flag)
		x = "1" + x;
	return x;
}

int main()
{
	std::string x;
	std::string y;
	while(cin>>x>>y)
    {
		cout << "x:" << x <<endl;
		cout << "y:" << y <<endl;
		cout << "sum:" << add(x,y) <<endl;
	}
	return 0;
}