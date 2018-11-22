#include <stdio.h>

//将数组相应位置的两个数字交换
void swap(int arr[], int i, int j)
{
  int k = arr[i];
  arr[i] = arr[j];
  arr[j] = k;
}

//采用Gleen W. Rowe划分算法
//根据一个基准数，将数组分为基准数左边小于基准数，基准数右边大于或等于基准数的两部分
//返回值是基准数在数组中的下标
//这里选取数组元素的第0位作为基准数
//low为最低下标，high为最高下标
int partition_rowe(int arr[], int low, int high)
{
  //选取基准数
  int pivot = arr[low];
  int low_index = low;
  int i;
  for (i = low + 1; i <= high; ++i)
  {
    if (arr[i] < pivot)
    {
      //在序列中找到一个比pivot小的，就递增low_index
      low_index++;
      //如果i和low_index相等，则在i之前都不存在需要交换的比pivot大的数
      if (i != low_index)
        swap(arr, i, low_index);
    }
  }
  //low_index的位置就是pivot应处在的位置，low_index指向的总是比pivot小的数
  arr[low] = arr[low_index];
  arr[low_index] = pivot;
  return low_index;
}

void quick_sort(int arr[], int low, int high)
{
  //如果需要排序的序列的元素个数大于1
  if (high > low)
  {
    int pivot_pos = partition_rowe(arr, low, high);
    //左序列
    quick_sort(arr, low, pivot_pos - 1);
    //右序列
    quick_sort(arr, pivot_pos + 1, high);
  }
}

int main(int argc, char *argv[])
{
  int arr[] = {5, 8, 7, 6, 4, 3, 9};
  int i;
  for (i = 0; i < 7; ++i)
    printf("%d,", arr[i]);
  printf("\n");
  quick_sort(arr, 0, 6);
  for (i = 0; i < 7; ++i)
    printf("%d,", arr[i]);
  printf("\n");
  return 0;
}
