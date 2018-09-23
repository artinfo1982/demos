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
			//解析出host填入host变量，例如http://1.1.1.1:8080/abc，i=7，url=http..., url+i=1.1..., strchr(url + i, ':') - url - i为1.1.1.1的长度 url
	   		strncpy(host, url + i, strchr(url + i, ':') - url - i);
	   		bzero(tmp, 10);
			//将端口解析出来，存入tmp
	   		strncpy(tmp, index(url + i, ':') + 1, strchr(url + i, '/') - index(url + i, ':') - 1);
	   		proxyport = atoi(tmp);
	   		if (proxyport == 0)
				proxyport = 80;
   		}
		//url不包含':'，例如http://1.1.1.1/abc，也是合法的，默认端口为80
		else
   		{
			/*
			#include <string.h>
			int strcspn(char *str, char *accept);
			strcspn计算字符串str中开头连续有几个字符都不属于字符串accept
			*/
     			strncpy(host, url + i, strcspn(url + i, "/"));
   		}
		//在request的最后加上URI部分，例如url=http://1.1.1.1:8080/abc，则在当前request的最后加上/abc
   		strcat(request + strlen(request), url + i + strcspn(url + i, "/"));
  	}
	//如果有proxy，则直接加上proxy的url
	else
		strcat(request,url);
	strcat(request, " HTTP/1.1\r\n");
  	if (proxyhost == NULL)
  	{
		strcat(request, "Host: ");
		strcat(request, host);
		strcat(request, "\r\n");
  	}
  	if (force_reload && proxyhost != NULL)
  	{
		strcat(request, "Pragma: no-cache\r\n");
  	}
	strcat(request, "Connection: close\r\n");
	if (method == METHOD_GET)
	{
		strcat(request, "Content-Length: 0\r\n");
		strcat(request, "\r\n");
	}
	else
	{
		body_len = strlen(req_body);
		sprintf(str_body_len, "%d", body_len);
		strcat(request, "Content-Length: ");
		strcat(request, str_body_len);
		strcat(request, "\r\n\r\n");
		strcat(request, req_body);
	}
}

static int bench(void)
{
	int i, j, k;	
	pid_t pid = 0;
	FILE *f;

  	/* check avaibility of target server */
  	i = Socket(proxyhost == NULL ? host : proxyhost, proxyport);
	if (i < 0)
	{ 
		fprintf(stderr, "\nConnect to server failed. Aborting benchmark.\n");
		return 1;
	}
	close(i);
	/*
	管道，是Linux支持的UNIX最初IPC形式之一，管道是半双工的，数据只能向一个方向流动。
	当需要实现客户端和服务器双向交互时，需要建立两个管道。
	管道只能用于父子进程之间，或者兄弟进程（由同一个父进程创建出来）之间的通信。
	管道是一种文件，但只存在于内存中。
	一个进程向管道中写的内容被管道另一端的进程读出，写入的内容每次都添加在管道缓冲区的末尾，并且每次都是从缓冲区的头部读出数据。
	管道的一端为读端，一端为写端，读端只能用于读，写端只能用于写，如果混用会报错。
	pipe(int fd[2])本身创建的管道两端都在一个进程中，没有实际意义，一般是将该进程fork后成为父子进程再使用管道。
	本程序中使用管道，主要是为了实现在父子进程之间传递统计信息（成功数、失败数等），便于在父进程中打印
	*/
	if (pipe(mypipe))
	{
		perror("pipe failed.");
		return 3;
	}

  	/* fork childs */
  	for (i = 0; i < clients; i++)
  	{
		pid = fork();
		if (pid <= (pid_t)0)
		{
			/* child process or error*/
			//此处sleep的作用是将父进程睡觉，目的是每次让子进程优先运行
			sleep(1);
			break;
		}
  	}
  	if (pid < (pid_t)0)
	{
		fprintf(stderr, "problems forking worker no. %d\n", i);
		perror("fork failed.");
		return 3;
	}

	//子进程
	if (pid == (pid_t)0)
	{
		if(proxyhost == NULL)
			benchcore(host, proxyport, request);
		else
			benchcore(proxyhost, proxyport, request);
		/*
		管道文件，不能直接用fopen打开，得到一个文件描述符，才能使用其他函数对其进行I/O
		w表示覆盖写，而w+表示追加写
		*/
	 	f = fdopen(mypipe[1], "w");
	 	if (f == NULL)
	 	{
			perror("open pipe for writing failed.");
			return 3;
	 	}
		//正常情况下，一个子进程会向管道写三个参数
	 	fprintf(f, "%d %d %d\n", speed, failed, bytes);
	 	fclose(f);
	 	return 0;
  	}
	//父进程
	else
  	{
		//从管道的读端读取数据
		f = fdopen(mypipe[0], "r");
	  	if(f == NULL) 
	  	{
			perror("open pipe for reading failed.");
			return 3;
	  	}
		/*
		#include <stdio.h>
		int setvbuf(FILE *stream, char *buf, int type, unsigned size);
		setvbuf设定文件流的缓冲区，type的取值说明如下：
		_IOFBF（满缓冲）：当缓冲区为空时，从流读入数据，或当缓冲区满时，向流写入数据
		_IOLBF（行缓冲）：每次从流中读入一行数据或向流中写入一行数据
		_IONBF（无缓冲）：直接从流中读入数据或直接向流中写入数据，而没有缓冲区
		*/
		//不使用文件流缓冲，直接从管道I/O
	  	setvbuf(f, NULL, _IONBF, 0);
	  	speed = 0;
          	failed = 0;
          	bytes = 0;
		
	  	while(1)
	  	{
			//fscanf是从一个流中格式化读入数据，类似于scanf，scanf是从终端输入，fscanf是从流读取
			//正常情况下，一个子进程会向管道写三个参数，父进程也会收到三个参数
			pid=fscanf(f, "%d %d %d", &i, &j, &k);
		  	if (pid < 2)
                  	{
                       		fprintf(stderr, "Some of our childrens died.\n");
                       		break;
                  	}
		  	speed += i;
		  	failed += j;
		  	bytes += k;
			//把所有client都统计，直到最后一个client
		  	if(--clients == 0)
				break;
	  	}
	  	fclose(f);
  		printf("\nSpeed=%d pages/min, %d bytes/sec.\nRequests: %d susceed, %d failed.\n", 
			(int)((speed+failed)/(benchtime/60.0f)),
			(int)(bytes/(float)benchtime),
		  	speed,
		  	failed);
  	}
  	return i;
}

void benchcore(const char *host, const int port, const char *req)
{
	int rlen;
	char buf[READ_BUF_SIZE];
	int s, i;
	/*
	#include <signal.h>
	struct sigaction {
		void (*sa_handler)(int);
		void (*sa_sigaction)(int, siginfo_t *, void *);
		sigset_t sa_mask;
		int sa_flags;
		void (*sa_restorer)(void);
	};
	*/
	//信号定义结构体
	struct sigaction sa;

	/* setup alarm signal handler */
	sa.sa_handler = alarm_handler;
	sa.sa_flags = 0;
	//如果收到定时器到时的信号，该进程就退出
	if (sigaction(SIGALRM, &sa, NULL))
		exit(3);
	alarm(benchtime);
 	rlen = strlen(req);
 nexttry:
	//死循环收发消息，直至进程退出
	while(1)
 	{
		//计时器到时
		if (timerexpired)
		{
			if (fail > 0)
				//计时器到时引起的最后一次失败已没有意义，删除
				fail--;
			return;
		}
    		s = Socket(host, porfail t);
		//创建socket失败
    		if (s < 0)
		{
			fail++;
			continue;
		} 
    		if (rlen != write(s, req, rlen))
		{
			fail++;
			close(s);
			continue;
		}
		//force=0强制需要等待服务器返回，force=1不等待服务器返回直接关闭socket
    		if (force == 0)
    		{
            		/* read all available data from socket */
	    		while(1)
	    		{
              			if (timerexpired)
					break; 
	      			i = read(s, buf, READ_BUF_SIZE);
	      			if (i < 0)
              			{ 
                 			fail++;
                	 		close(s);
                 			goto nexttry;
              			}
	       			else if (i == 0)
					break;
		       		else
			       		bytes += i;
	    		}
    		}
		//直接关闭socket
    		if (close(s))
		{
			fail++;
			continue;
		}
    		success++;
 	}
}
