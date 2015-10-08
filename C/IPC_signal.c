#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

/*
* this is a demo to show multi processes IPC with pipe
* 1 --> 2, 1 --> 3, 1 write 20 bytes, 2 read 10 bytes, then 3 read 10 bytes
*/

#define MYSIG 62

static void signal_handler(int sig)
{
	switch (sig)
	{
		case MYSIG:
			{
				printf("2 get signal %d from 1\n", MYSIG);
				break;
			}		
		default:
			{
				printf("no signal\n");
				break;
			}		
	}
}

void child_do()
{
	if (signal(MYSIG, signal_handler) == SIG_ERR)
	{
		printf("can not catch %d\n", MYSIG);
		exit(1);
	}
	pause();
	exit(0);
}

void father_do(int pid)
{
	int status;
	sleep(3);
	kill(pid, MYSIG);
	waitpid(pid, &status, 0);
}

int main(int argc,char *argv[])
{
	int pid;

	// 1-->2
	switch(pid = fork())
	{
		case -1:
			{
				printf("fork error\n");
				exit(1);
			}
		case 0:
			{
				child_do();
				break;
			}
		default:
			{
				printf("1 --> 2, 1 id=%d, 2 id=%d\n", getpid(), pid);
				father_do(pid);
				break;
			}		
	}

	return 0;
}
