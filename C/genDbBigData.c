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
  
  unsigned long loop = atol(argv[1]);
  unsigned long u_len = atol(argv[2]);
  unsigned long u_num = atol(argv[3]);
  unsigned long all_unit_len = (u_len + 1) * u_num;
  char *file = argv[4];
  unsigned long i;
  unsigned int index_len;
  unsigned long row_len;
  unsigned int fd = -1, ret = -1;
  unsigned long offset = 0L;
  
  if ((fd = open(file, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)) < 0)
  {
    printf("Can not open file: %s\n", file);
    return 1;
  }
  
  char *p_unit = (char *)malloc((u_len + 2) * sizeof(char));
  memset(p_unit, 0x0, u_len + 2);
  memcpy(p_unit, SEPARATOR, 1);
  for (i=1; i<u_len+1; i++)
    memcpy(p_unit + i, STR_UNIT + 1);
  
  char *p_all_unit = (char *)malloc((all_unit_len + 1) * sizeof(char));
  memset(p_all_unit, 0x0, all_unit_len + 1);
  
  for (i=0; i<u_num; i++)
    memcpy(p_all_unit + (u_len + 1) * i, p_unit, u_len + 1);
  
  char *p_row = (char *)malloc((all_unit_len + 256) * sizeof(char));
  
  for (i=1; i<=loop; i++)
  {
    index_len = digit(i);
    memset(p_row, 0x0, all_unit_len + 256);
    sprintf(p_row, "%ld%s\n", i, p_all_unit);
    row_len - index + all_unit_len + 1;
    if ((ret = pwrite(fd, p_row, row_len, offset)) < 0)
    {
      prinntf("pwrite error, pls check!\n");
      return 1;
    }
    offset += row_len;
    if (i % (loop / 10) == 0)
      printf("finished %ld\n", i);
  }
  close(fd);
  free(p_unit);
  free(p_all_unit);
  free(p_row);
  p_unit = NULL;
  p_all_unit = NULL;
  p_row = NULL;
  
  return 0;
}
