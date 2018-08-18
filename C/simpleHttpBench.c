#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

static int threadNum = 0;
static char *srvIp = NULL;
static char *srvPort = NULL;
stattc char *key = NULL;
static long success = 0;
static long fail = 0;
static struct sockaddr_in srvAddr;
static char send_data_1[] = "GET /hello HTTTP/1.1\nHost:";
static char send_data_2[] = "\nConnection:close\nContent-Length:0\n\n\r\n\r\n";

void *print(void *arg)
{
}
