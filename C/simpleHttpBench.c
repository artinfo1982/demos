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
  long start, end;
  for (;;)
  {
    start = success;
    usleep(4000000);
    end = success;
    printf("total success=%ld, fail=%ld, tps=%d, errRate=%.2f%%\n", success, fail, (end - start) >> 2, 100 * fail / (success + fail));
  }
}

void *work(void *arg)
{
  int sfd = -1;
  char buffer[1024];
  char *res = NULL;
  
  struct iovec iov[5];
  iov[0].iov_base = send_data_1;
  iov[0].iov_len = strlen(send_data_1);
  iov[1].iov_base = srvIp;
  iov[1].iov_len = strlen(srvIp);
  iov[2].iov_base = ":";
  iov[2].iov_len = 1;
  iov[3].iov_base = srvPort;
  iov[3].iov_len = strlen(srvPort);
  iov[4].iov_base = send_data_2;
  iov[4].iov_len = strlen(send_data_2);
  
  signal(SIGPIPE, SIG_IGN);
  
  for (;;)
  {
    if ((sfd = socket(AF_INET, SOCK_STREAM, 0)) < 0 )
    {
      printf("create socket failed\n");
      exit(1);
    }
    if ((connect(sfd, (struct sockaddr *)(&srvAddr), sizeof(struct sockaddr))) < 0)
    {
      __sync_fetch_and_add(&fail, 1);
      continue;
    }
    if ((writev(sfd, iov, 5)) < 0)
    {
      close(sfd);
      __sync_fetch_and_add(&fail, 1);
      continue;
    }
    if ((recv(sfd, buffer, 1024, 0)) < 0)
    {
      close(sfd);
      __sync_fetch_and_add(&fail, 1);
      continue;
    }
  }
}
