#include <stdio.h>
#include <unistd.h>
#include <string.h>

/**
 * 字符串的完全就地逆序
 * 原理：对两个存储域交换使用异或，可以达到交换这两个存储域的值的目的，而不需要额外的变量参与
 * 例如：原始的字符串为abcde，完全逆序后是edcba
 */
char * fullReverseString(char * str)
{
        size_t len, i;
        char *head=NULL, *tail=NULL;
        len = strlen(str);
        //head指针指向字符串的头
        head = str;
        //tail指针指向字符串最后紧靠'\0'的那个字符
        tail = str + len -1;
        //head指针逐个向后，tail指针逐个向前，直至相遇结束
        for (i=0; i<len/2; i++)
        {
                //head指针和tail指针相遇
                if ((head+i) == (tail-i))
                        break;
                //异或操作交换任意两个字符
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
        printf("%s\n", fullReverseString(argv[1]));
        return 0;
}
