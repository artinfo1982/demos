#include <iostream>
#include<string>
using namespace std;

int compare(string &num1,string &num2)
{
    int l1 = num1.size();
    int l2 = num2.size();
    if(l1>l2)
        return 1;
    else if(l1<l2)
        return 2;
    else 
        for(int i=0;i<l1;i++)
          {
            if(num1[i]>num2[i])
                return 1;
            else if(num1[i]<num2[i])
                return 2;
           } 
    return 0;
}

string minus(string &num1,string &num2)
{
    int len1= num1.size();
    int len2= num2.size();
    int flag = 0;
    int temp=0;
    string result;
    for(int i=0;i<len2;i++)
    {
        temp=num1[len1-1-i]-num2[len2-1-i]-flag;
        if(temp<0){
            flag=1;
            temp = temp+10;
            result.push_back(temp+'0');
        }
        else {
            flag=0;
            result.push_back(temp+'0');     
        }
    }
    for(int i=len1-len2-1;i>=0;i--)
    {
        temp = num1[i]-'0'-flag;
        if(temp<0)
        {
            flag =1;
            result.push_back(temp+10+'0');      
        }
        else
        {
            flag=0;
            result.push_back(temp+'0');     

        }
    }
    int len3 = result.size();
    int num=0;
    for(int i=0;i<len3;i++)
        if(result[i]==0)
            num++;
        else
            break;
    if(num>0)
        result.erase(len3-num);
    num = result.size();
    char c;
    for(int i=0;i<num/2;i++)
    {
        c = result[i];
        result[i]=result[num-1-i];
        result[num-1-i] = c;
    }
    return result;
}

void main()
{
    string num1,num2;
    cin>>num1>>num2;
    int flag=0;
    string result;
    flag = compare(num1,num2);
    if(flag==0)
        {
        result='0';
    }
    if(flag==1)
        result = minus(num1,num2);
    if(flag==2){
        result.push_back('-');
        result.append(minus(num2,num1)); 
    }
    cout<<result;
    return ;
}