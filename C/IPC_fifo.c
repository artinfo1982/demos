#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

/*
* this is a demo to show multi processes IPC with fifo
* 1 --> 2, 1 --> 3, 2 write 3 bytes, 3 write 3 bytes, then 1 read 6 bytes
* fifo, read should be opened first, then write
*/

void child_1_do()
{
	int fd = -1;
	char * data = "AAA";
	sleep(5);
	if ((fd = open("/tmp/fifo", O_WRONLY | O_NONBLOCK)) < 0)
	{
		printf("2 open fifo error\n");
		exit(1);
	}
	if (write(fd, data, 3) < 3)
	{
		printf("2 send error\n");
		exit(1);
	}
	close(fd);
	exit(0);
}

void child_2_do()
{
	int fd = -1;
	char * data = "BBB";
	sleep(10);
	if ((fd = open("/tmp/fifo", O_WRONLY | O_NONBLOCK)) < 0)
	{
		printf("3 open fifo error\n");
		exit(1);
	}
	if (write(fd, data, 3) < 3)
	{
		printf("3 send error\n");
		exit(1);
	}
	close(fd);
	exit(0);
}

void father_do(int pid1, int pid2)
{
	int status, fd = -1;
	char buf[7];
	memset(buf, 0x0, sizeof(buf));
	if ((fd = open("/tmp/fifo", O_RDONLY | O_NONBLOCK)) < 0)
	{
		printf("1 open fifo error\n");
		exit(1);
	}
	sleep(20);
	if (read(fd, buf, 6) < 6)
	{
		printf("1 recv error\n");
		exit(1);
	}
	buf[7] = '\0';
	close(fd);
	printf("1 recv data from 2 and 3, data=%s\n", buf);
	waitpid(pid1, &status, 0);
	waitpid(pid2, &status, 0);
}

int main(int argc,char *argv[])
{
	int pid1, pid2;

	if (mkfifo("/tmp/fifo", 0777) < 0)
	{
		printf("mkfifo error\n");
		exit(1);
	}

	// 1-->2
	switch(pid1 = fork())
	{
		case -1:
			{
				printf("fork error\n");
				exit(1);
			}
		case 0:
			{
				child_1_do();
			}
		default:
			{
				printf("1 --> 2, 1 id=%d, 2 id=%d\n", getpid(), pid1);
				// 1 --> 3
				switch (pid2 = fork())
				{
					case -1:
						{
							printf("fork error\n");
							exit(1);
						}
					case 0:
						{
							child_2_do();
						}
					default:
						printf("1 --> 3, 1 id=%d, 3 id=%d\n", getpid(), pid2);
						break;
				}
				break;
			}		
	}
	
	father_do(pid1, pid2);

	unlink("/tmp/fifo");
	remove("/tmp/fifo");
	return 0;
}
