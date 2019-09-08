#include <stdio.h>
#include <stdlib.h>

/*
* 快速排序的一种，只允许遍历一次任意的整数数组，即完成排序。
* 算法思路：利用双向链表，对于数组，自左而右，不断比较新值应该插在已排好序的链表的何处。
* 该算法优点：不需要任何交换操作，而一般的排序算法（冒泡、快速）都不可避免会涉及数组元素的交换次序。
*/

struct _Node
{
    int n;
    struct _Node *pre;
    struct _Node *next;
};

typedef struct _Node Node;

int main(int argc, char *argv[])
{
    int a[] = {3, 7, 5, 2, 8, 1, 9, 4, 2, 14, -1, 90, 23, 10, 82, 9, 3};
    int i;
        
    Node *root = (Node*)malloc(sizeof(Node));
    root->n = a[0];
    root->pre = NULL;
    root->next = NULL;

    Node *tail = root;

    for (i=1; i<17; i++)
    {
        Node *new = (Node*)malloc(sizeof(Node));
        new->n = a[i];
        if (new->n >= tail->n)
        {
            tail->next = new;
            new->pre = tail;
            new->next = NULL;
            tail = new;
        }
        else
        {
            Node *p1 = tail;
            while (new->n < p1->n)
            {
                if (NULL != p1->pre)
                    p1 = p1->pre;
                else
                    break;
            }
            if ((NULL == p1->pre) && (new->n < p1->n))
            {
                new->next = p1;
                p1->pre = new;
                new->pre = NULL;
                root = new;          
            }
            else
            {
                Node *p2 = p1->next;
                p1->next = new;
                new->next = p2;
                new->pre = p1;
                p2->pre = new;
            }
        }
    }

    Node *t = root;
    while (t->next != NULL)
    {
        printf("%d\n", t->n);
        t = t->next;
    }
    printf("%d\n", t->n);
    return 0;
}
