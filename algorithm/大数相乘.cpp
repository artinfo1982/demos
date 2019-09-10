#include<iostream>
#include<deque>
#include<sstream>

std::string BigNumMultiply(std::string s1,std::string s2)
{
    //记录最终结果
    std::string res="";
    //使用deque是因为出现进位时可以在队列前插入数据，效率比vector高，大小设为最小
    std::deque<int> vec(s1.size()+s2.size()-1,0);
    for(int i=0;i<s1.size();++i)
    {
        for(int j=0;j<s2.size();++j)
            vec[i+j]+=(s1[i]-'0')*(s2[j]-'0');//记录相乘结果
    }
    //进位处理
    int addflag=0;
    //倒序遍历，是因为最左边的值为最高位，最右边的值在最低位，进位运算要从低位开始
    for(int i=vec.size()-1;i>=0;--i)
    {
        int temp=vec[i]+addflag;//当前值加上进位值
        vec[i]=temp%10;//当前值
        addflag=temp/10;//进位值
    }
    //如果有进位，将进位加到队列头部
    while(addflag!=0)
    {
        int t=addflag%10;
        vec.push_front(t);
        addflag/=10;
    }
    for(auto c:vec)
    {
        std::ostringstream ss;
        ss<<c;
        res=res+ss.str();
    }
    return res;
}

int main()
{
    std::string str1,str2;
    while(std::cin>>str1>>str2)
    {
        std::cout<<str1<<"*"<<str2<<"="<<std::endl;
        std::cout<<BigNumMultiply(str1,str2)<<std::endl;
    }
    return 0;
}