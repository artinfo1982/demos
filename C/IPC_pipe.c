#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/*
* this is a demo to show multi processes IPC with pipe
* 1 --> 2, 1 --> 3, 1 write 20 bytes, 2 read 10 bytes, then 3 read 10 bytes
*/

void child_1_do(int fd)
{
	char buf[11];
	printf("2 sleep 10 seconds to recv\n");
	sleep(10);
	if (read(fd, buf, 10) < 10)
	{
		printf("2 recv error\n");
		exit(1);
	}
	buf[11] = '\0';
	printf("2 recv 10 bytes, data=%s\n", buf);
	exit(0);
}

void child_2_do(int fd)
{
	char buf[11];
	printf("3 sleep 30 seconds to recv\n");
	sleep(30);
	if (read(fd, buf, 10) < 10)
	{
		printf("3 recv error\n");
		exit(1);
	}
	buf[11] = '\0';
	printf("3 recv 10 bytes, data=%s\n", buf);
	exit(0);
}

void father_do(int pid1, int pid2, int fd)
{
	int status;
	char * w_data = "0123456789abcdefghij";
	if (write(fd, w_data, 20) < 20)
	{
		printf("1 send error\n");
		exit(1);
	}
	printf("1 send 20 bytes\n");
	waitpid(pid1, &status, 0);
	waitpid(pid2, &status, 0);
	exit(0);
}

int main(int argc,char *argv[])
{
	int pid1, pid2;
	// fd[0] for read, fd[1] for write
	int fd[2];

	pipe(fd);

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
				close(fd[1]);
				child_1_do(fd[0]);
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
							close(fd[1]);
							child_2_do(fd[0]);
						}
					default:
						printf("1 --> 3, 1 id=%d, 3 id=%d\n", getpid(), pid2);
						break;
				}
				break;
			}		
	}
	
	close(fd[0]);
	father_do(pid1, pid2, fd[1]);
	return 0;
}
