/**
 * D状态进程一般kill -9无法起作用，本程序将进程的状态由D-->T(TASK_STOPPED)，就可以kill了
 * 
 * 编译方法：
 * Makefile的内容：
 * obj-m := proc_d2t.o
 * all:
 *  $(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
 * clean:
 *  $(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
 * 执行make，会生成一个proc_d2t.ko文件
 * 
 * 使用方法：
 * insmod ./proc_d2t.ko pid=xxxx
 * rmmod proc_d2t.ko  (卸载内核模块)
 *
 * 注意，在4.x的内核中，需要添加 #include <linux/sched/signal.h>，其他内核是否需要添加，请查阅for_each_process在哪个头文件中定义
 */

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched/signal.h>
#include <linux/sched.h>

MODULE_AUTHOR("ChenDong");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("This module make process from D to T, then you can kill it");

static int pid = -1;
module_param(pid, int, S_IRUGO);

static int __init init(void)
{
    struct task_struct *p;
    for_each_process(p)
    {
        if(p->pid == pid)
        {
            p->state = TASK_STOPPED;
            return 0;
        }
    }
    return 0;
}

static void __exit exit(void)
{
}

module_init(init);
module_exit(exit);
