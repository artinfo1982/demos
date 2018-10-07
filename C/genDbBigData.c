/*
快速创建指定格式的txt文件，用于导入数据库，比如hive、oracle等。
数据格式：id,列1,列2,...
*/

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>

#define STR_UNIT "1"
#define SEPARATOR ","

int digit(long number)
{
  long n = number;
  int count = 0;
  while (n != 0)
  {
    n /= 10;
    count++;
  }
  return count;
}

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    printf("Params Error! Usage: %s loopNumber unitLength unitNumber outputFile\n", argv[0]);
    printf("Example: %s 100000000 100 100 /home/aaa/out.txt\n", argv[0]);
    return 1;
  }
  
}
