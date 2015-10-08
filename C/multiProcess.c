#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>

/*
* this is a demo to show multi processes on Linux
*/

void process()
{
	printf("i am child, my id=%d\n", getpid());
	sleep(60);
	exit(0);
}

int main(int argc,char *argv[])
{
	int i, status;
	int pid[2];

	for (i=0; i<2; i++)
	{
		switch(pid[i] = fork())
		{
			case -1:
			{
				printf("fork error\n");
				exit(1);
			}
			// child
			case 0:
			{
				process();
			}
			// father
			default:
			{
				printf("i am father, my id=%d, my child id=%d\n", getpid(), pid[i]);
				// WNOHANG is none blocking, and 0 is blocking
				waitpid(pid[i], &status, WNOHANG);
				exit(0);
			}
		}
	}
	sleep(60);
	return 0;
}
