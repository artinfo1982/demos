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
	if (argc == 1)
	{
		usage();
		return 2;
	}
	
	/*
	getopt_long，解析命令行参数
	函数定义原型：
	int getopt_long(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex);
	argc, argv不解释
	两个冒号之间的参数，表示后面必须强制带上参数，比如-p xxx
	选项后面跟的参数是optarg，举例-t 30，这个30就是optarg，只是默认是字符串类型
	optind是指向不能解析的第一个参数位置，比如./test -f -r http://1.1.1.1:2222/，其中optind就指向URL的起始位置，./test从0开始，此时optind=3，argc=4
	*/
	while ((opt = getopt_long(argc, argv, "frt:p:c:d:?h", long_options, &options_index)) != EOF)
 	{
		switch(opt)
 		{
			case  0:
				break;
			case 'f':
				force = 1;
				break;
			case 'r':
				force_reload = 1;
				break; 
			case 't':
				benchtime = atoi(optarg);
				break;	     
			case 'p':
				/*
				解析代理信息，server:port
				strchr，查找一个字符串在另一个字符串中末次出现的位置，并返回从字符串中的这个位置起，一直到字符串结束的所有字符
				比如-p http://1.1.1.1:2222，tmp=2222，由于80端口默认可以不写，所以可能出现-p http://1.1.1.1，此时tmp就是NULL
				*/
	     			tmp = strrchr(optarg, ':');
	     			proxyhost = optarg;
				//只有hostname
	     			if (tmp == NULL)
		     			break;
				//缺少hostname，比如-p :2222
	     			if (tmp == optarg)
	     			{
		     			fprintf(stderr, "Error in option --proxy %s: Missing hostname.\n", optarg);
		     			return 2;
	    	 		}
				//缺少port，比如-p http://1.1.1.1:
	     			if(tmp==optarg+strlen(optarg)-1)
	     			{
		     			fprintf(stderr, "Error in option --proxy %s Port number is missing.\n", optarg);
		     			return 2;
	     			}
	     			*tmp = '\0';
	     			proxyport =atoi(tmp + 1);
				break;
   			case ':':
   			case 'h':
   			case '?':
				usage();
				return 2;
				break;
   			case 'c':
				clients = atoi(optarg);
				break;
			case 'd':
				if (strlen(optarg) > REQUEST_BODY_SIZE)
				{
					fprintf(stderr, "ERROR: Request body length more than %d!\n", REQUEST_BODY_SIZE);
		     			return 2;
				}
				req_body = malloc(REQUEST_BODY_SIZE * sizeof(char));
				memset(req_body, 0x0, REQUEST_BODY_SIZE);
				req_body = optarg;
				break;
  		}
 	}
	
	//缺少URL，例如./test -f -r, argc=3, optind=3
	if (optind == argc)
	{
		fprintf(stderr,"webbench: Missing URL!\n");
		usage();
		return 2;
	}
	
	//如果不填并发数，默认一个并发
	if (clients == 0)
		clients = 1;
	//如果不填测试时长，默认测试时长60秒
	if (benchtime == 0)
		benchtime = 60;
	build_request(argv[optind]);
 	//print bench info
 	printf("\nBenchmarking: ");
 	switch(method)
 	{
		case METHOD_GET:
			printf("GET");
			break;
		case METHOD_POST:
			printf("POST");
			break;
		case METHOD_PUT:
			printf("PUT");
			break;
		case METHOD_DELETE:
			printf("DELETE");
			break;
		default:
			break;
 	}
 	printf(" %s", argv[optind]);
	printf("\n");
 	printf("%d clients, running %d sec", clients, benchtime);
 	if (force)
		printf(", early socket close");
 	if (proxyhost != NULL)
		printf(", via proxy server %s:%d", proxyhost, proxyport);
 	if (force_reload)
		printf(", forcing reload");
 	printf(".\n");
 	return bench();
}

void build_request(const char *url)
{
	char tmp[10];
	int i;
	int body_len = 0;
	char str_body_len[8];

  	bzero(host, MAXHOSTNAMELEN);
  	bzero(request, REQUEST_SIZE);
	bzero(str_body_len, 8);

  	switch(method)
  	{
		case METHOD_GET:
			strcpy(request, "GET");
			break;
		case METHOD_POST:
			strcpy(request, "POST");
			break;
		case METHOD_PUT:
			strcpy(request, "PUT");
			break;
		case METHOD_DELETE:
			strcpy(request, "DELETE");
			break;
		default:
			break;
  	}
		  
  	strcat(request, " ");

  	if (NULL == strstr(url, "://"))
  	{
		fprintf(stderr, "\n%s: is not a valid URL.\n",url);
		exit(2);
  	}
  	if (strlen(url) > REQUEST_URL_LENGTH)
	{
		fprintf(stderr, "URL is too long, more than %d bytes.\n", REQUEST_URL_LENGTH);
		exit(2);
	}
 	if (proxyhost == NULL)
	{
		if (0 != strncasecmp("http://", url, 7)) 
		{
			fprintf(stderr, "URL must begin with 'http://'\n");
			exit(2);
		}
	}
	//获取url的hostname的起始位置，例如http://1.1.1.1:8080，则i=7（4-0+3），指向1.1.1.1的首地址
  	i = strstr(url, "://") - url + 3;
	//host必须以'/'结尾，目的是便于解析端口信息，比如http://1.1.1.1:8080/，最后一个/的目的只是为了便于后续的代码解析8080
  	if (strchr(url + i, '/') == NULL)
	{
		fprintf(stderr, "URL must ends with '/'.\n");
		exit(2);
	}
  	if (proxyhost == NULL)
  	{
		/*
		#include <string.h>
		char *index(const char *s, int c);
		index函数返回字符串s中第一个出现c的地址，字符串结束字符（NULL）也视为字符串一部分
		*/
		//如果url包含':'，并且':'出现在url包含的'/'的前面
		if (index(url + i, ':') != NULL && index(url + i, ':') < index(url + i, '/'))
   		{
	   strncpy(host,url+i,strchr(url+i,':')-url-i);
	   bzero(tmp,10);
	   strncpy(tmp,index(url+i,':')+1,strchr(url+i,'/')-index(url+i,':')-1);
	   /* printf("tmp=%s\n",tmp); */
	   proxyport=atoi(tmp);
	   if(proxyport==0) proxyport=80;
   } else
   {
     strncpy(host,url+i,strcspn(url+i,"/"));
   }
   // printf("Host=%s\n",host);
   strcat(request+strlen(request),url+i+strcspn(url+i,"/"));
  } else
  {
   // printf("ProxyHost=%s\nProxyPort=%d\n",proxyhost,proxyport);
   strcat(request,url);
  }
  if(http10==1)
	  strcat(request," HTTP/1.0");
  else if (http10==2)
	  strcat(request," HTTP/1.1");
  strcat(request,"\r\n");
  if(http10>0)
	  strcat(request,"User-Agent: WebBench "PROGRAM_VERSION"\r\n");
  if(proxyhost==NULL && http10>0)
  {
	  strcat(request,"Host: ");
	  strcat(request,host);
	  strcat(request,"\r\n");
  }
  if(force_reload && proxyhost!=NULL)
  {
	  strcat(request,"Pragma: no-cache\r\n");
  }
  if(http10>1)
	  strcat(request,"Connection: close\r\n");
  /* add empty line at end */
  if(http10>0) strcat(request,"\r\n"); 
  // printf("Req=%s\n",request);
}
