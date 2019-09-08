#include <stdio.h>
#include <unistd.h>
#include <string.h>

/*
* 部分字符串翻转，例如12 345 6789，翻转后得到6789 345 12
*/

// 通用字符串翻转函数
char * reverseString(char * str, size_t len)
{
        size_t i;
        char *head=NULL, *tail=NULL;
        head = str;
        tail = str + len -1;
        for (i=0; i<len/2; i++)
        {
                if ((head+i) == (tail-i))
                        break;
                *(head+i) ^= *(tail-i);
                *(tail-i) ^= *(head+i);
                *(head+i) ^= *(tail-i);
        }
        return str;
}

int main(int argc, char *argv[])
{
        if (argc < 2)
        {
                printf("intput error, string can not null\n");
                _exit(1);
        }
        // 先做一次字符串的整体完全翻转
        reverseString(argv[1], strlen(argv[1]));
        
        char * p1 = argv[1];
        char * p2 = argv[1];

        while (*p2 != '\0')
        {
                if (*p2 == ' ')
                {
                        // 以空格为分隔，将每一个子串进行二次翻转
                        reverseString(p1, p2-p1);
                        p1 = p2++;
                        p1++;
                }
                else
                {
                        p2++;  
                }              
        }
        // p2指向字符串的'\0'，对最后一个子字符串进行二次翻转
        reverseString(p1, p2-p1);
        printf("%s\n", argv[1]);
        return 0;
}
