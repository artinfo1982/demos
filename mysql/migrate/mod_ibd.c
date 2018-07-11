/*
本程序专用于mysql的快速迁移。
假设mysql迁移从A到B，一般有如下几个步骤：
(1)B创建和A同名的表空间和表
(2)复制A的所有*.ibd文件到临时目录
(3)修改临时目录中所有*.ibd文件的第37-38字节和第41-42字节内容，与A相应的ibd文件的第37-38字节内容保持一致
(4)将临时目录中所有的*.ibd文件复制到B中的相应位置
(5)重启B

gcc mod_ibd.c -o mod_ibd -O3
*/

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("ERROR, params error\n");
        printf("Usage: %s src_ibd_file_name des_ibd_file_name\n", argv[0]);
        exit(1);
    }
    char *in_ibd = argv[1];
    char *out_ibd = argv[2];
    char *p1 = (char*)malloc(1 * sizeof(char));
    char *p2 = (char*)malloc(1 * sizeof(char));
    memset(p1, 0x0, 1);
    memset(p2, 0x0, 1);
    int fd = -1;
    if ((fd = open(in_ibd, O_RDONLY)) < 0)
    {
        printf("ERROR, can not open ibd file: %s\n", in_ibd);
        exit(1);
    }
    lseek(fd, 36, SEEK_SET);
    read(fd, p1, 1);
    lseek(fd, 37, SEEK_SET);
    read(fd, p2, 1);
    close(fd);

    if ((fd = open(out_ibd, O_WRONLY)) < 0)
    {
        printf("ERROR, can not open ibd file: %s\n", out_ibd);
        exit(1);
    }
    lseek(fd, 36, SEEK_SET);
    write(fd, p1, 1);
    lseek(fd, 37, SEEK_SET);
    write(fd, p2, 1);
    lseek(fd, 40, SEEK_SET);
    write(fd, p1, 1);
    lseek(fd, 41, SEEK_SET);
    write(fd, p2, 1);
    close(fd);

    free(p1);
    free(p2);
    p1 = NULL;
    p2 = NULL;

    return 0;
}
