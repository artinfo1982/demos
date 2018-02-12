/**
 * 将进程的状态由normal(S)-->Z状态，经过指定的时间后恢复
 * 
 * 编译方法：
 * Makefile的内容：
 * obj-m := proc_s2z.o
 * all:
 *  $(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
 * clean:
 *  $(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
 * 执行make，会生成一个proc_s2z.ko文件
 * 
 * 使用方法：
 * insmod ./proc_s2z.ko pid=xxxx delay=10  (表示Z状态持续10秒后恢复为S状态)
 * rmmod proc_s2z.ko  (卸载内核模块)
 */

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/delay.h>

MODULE_AUTHOR("ChenDong");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("This module make process from S to Z, then back to S");

static int pid = -1;
static int delay = -1;
module_param(pid, int, S_IRUGO);
module_param(delay, int, S_IRUGO);

static int __init init(void)
{
    struct task_struct *p;
    for_each_process(p)
    {
        if(p->pid == pid)
        {
            p->exit_state = EXIT_ZOMBIE;
            mdelay(delay * 1000);
            p->exit_state = TASK_INTERRUPTIBLE;
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
