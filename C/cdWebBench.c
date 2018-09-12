/*
修改自webbench
修改点：
1.只支持http 1.1
2.增加对POST、PUT、DELETE的支持
3.去除对TRACE、OPTIONS、HEAD的支持

使用方法：
gcc cdWebBench.c -o cdWebBench -O3
./cdWebBench -t 300 -c 10 --get http://192.168.1.1:8080/abc
./cdWebBench -t 300 -c 10 --post -d '{"a":"1"}' http://192.168.1.1:8080/abc
*/

#include <sys/types.h>
#include <sys/socket.h>
#include <fchtl.h>
#include <netinet.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/param.h>
#include <rpc/types.h>
#include <getopt.h>
#include <strings.h>
#include <time.h>
#include <signal.h>

/*
volatile的主要目的是在程序运行期，当需要该变量时，强制从内存中读取当前最新的值，而不是读缓存的旧值。
一般用于多进程编程，当一个变量会反复被很多进程读写。
timerexpired变量在本程序中的作用是标识该进程是否已经达到设定的测试时长，0表示时间没到，1表示时间到了
*/
volatile static int timerexpired = 0;
static int success = 0;
static int fail = 0;
static int bytes = 0;
//Allow: GET, POST, PUT, DELETE
#define METHOD_GET 0
#define METHOD_POST 1
#define METHOD_PUT 2
#define METHOD_DELETE 3
static int method = METHOD_GET;
static int clients = 1;
static int force = 0;
static int force_reload = 0;
static int proxyport = 80;
static char *proxyhost = NULL;
static int benchtime = 30;
/*
定义管道，用于父子进程之间的交互，在后续详细解释。
*/
static int mypipe[2];
//在<sys/param.h>中，MAXHOSTNAMELEN定义为256
static char host[MAXHOSTNAMELEN];
#define REQUEST_SIZE 4096
static char request[REQUEST_SIZE];
#define READ_BUF_SIZE 4096
#define REQUEST_BODY_SIZE 2048
#define REQUEST_URL_LENGTH 1500
static char *req_body;

/*
option结构体的定义如下：
struct option {
  const char *name; --- 选项名称
  int has_arg; --- 选项后面是否跟参数
  int *flag; --- 返回行为，NULL则返回val，如果指向一个变量，则将val赋值给变量
  int val; --- 值
};
has_arg有如下取值的可能：
no_argument表示后面不允许带参数，required_argument表示后面必须带参数
no_argument也可以用0代替，required_argument也可以用1代替，2表示既可以带也可以不带参数
*/
static const struct option long_options[] =
{
	{"force", no_argument, &force, 1},
  	{"reload", no_argument, &force_reload, 1},
  	{"time", required_argument, NULL, 't'},
  	{"help", no_argument, NULL, '?'},
  	{"get", no_argument, &method, METHOD_GET},
  	{"post", no_argument, &method, METHOD_POST},
  	{"put", no_argument, &method, METHOD_PUT},
  	{"delete", no_argument, &method, METHOD_DELETE},
  	{"proxy", required_argument, NULL, 'p'},
  	{"clients", required_argument, NULL, 'c'},
  	{"data", required_argument, NULL, 'd'},
  	{NULL, 0, NULL, 0}
};

//下面这个Socket函数取自 Virginia Tech Computing Center
int Socket(const char *host, int clientPort)
{
	int sock;
	unsigned long inaddr;
	struct sockaddr_in ad;
	struct hostent *hp;
	memset(&ad, 0, sizeof(ad));
	ad.sin_family = AF_INET;
	inaddr = inet_addr(host);
	//如果不是一个合法的ip地址，会返回INADDR_NONE
	if (inaddr != INADDR_NONE)
		memcpy(&ad.sin_addr, &inaddr, sizeof(inaddr));
	else
	{
		hp = gethostbyname(host);
		if (hp == NULL)
			return -1;
		memcpy(&ad.sin_addr, hp->h_addr, hp->h_length);
	}
	ad.sin_port = htons(clientPort);
	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0)
		return sock;
	if (connect(sock, (struct sockaddr *)&ad, sizeof(ad)) < 0)
		return -1;
	return sock;
}

static void benchcore(const char* host,const int port, const char *request);
static int bench(void);
static void build_request(const char *url);
//注册信号处理函数，此处是针对定时器到时的信号处理
static void alarm_handler(int signal)
{
	//标记计时器到时
	timerexpired = 1;
}

static void usage(void)
{
	fprintf(stderr,
	"cdWebBench [option]... URL\n"
	"  -f|--force\t\t\tDon't wait for reply from server.\n"
	"  -r|--reload\t\t\tSend reload request - Pragma: no-cache.\n"
	"  -t|--time <sec>\t\tRun benchmark for <sec> seconds. Default 30.\n"
	"  -p|--proxy <server:port>\tUse proxy server for request.\n"
	"  -c|--clients <n>\t\tRun <n> HTTP clients at once. Default one.\n"
	"  --get\t\t\t\tUse GET request method.\n"
        "  --post\t\t\tUse GET request method.\n"
        "  --put\t\t\t\tUse GET request method.\n"
        "  --delete\t\t\tUse GET request method.\n"
        "  -d|--data <string>\t\tSend data, which POST, PUT, DELETE needed\n"
	"  -?|-h|--help\t\t\tThis information.\n");
};

int main(int argc, char *argv[])
{
	int opt = 0;
	int options_index = 0;
	char *tmp = NULL;
	if(argc == 1)
	{
		usage();
		return 2;
	}
}
