#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "PlrIntoSrc.h"

using namespace cv;

//真实车牌对应的类别名，值对应关系可以参考如下的函数定义
//plate_licence_recognition/generate_licence_char/src/PlrChrGen.cpp中的class_parser函数
const int licence_chr_class[MAX_PLR_FILE_NUM][8] = {
  {12,47,46,61,41,33,37,0}, {12,42,37,40,37,44,32,0}, {2,43,34,35,33,34,37,0}, {13,44,65,50,38,40,39,0}, 
  {12,47,37,35,38,51,63,0}, {12,42,32,34,33,34,51,0}, {12,47,33,33,38,40,49,0}, {12,47,44,37,40,37,33,0}, 
  {12,42,40,65,44,38,38,0}, {2,49,38,32,41,33,37,0}, {2,43,39,37,33,37,35,0}, {11,43,43,32,34,34,41,0}, 
  {12,42,55,37,39,32,41,0}, {12,47,43,40,33,37,35,0}, {12,47,33,35,41,53,48,0}, {12,42,44,37,40,34,54,0}, 
  {11,4742,52,40,36,41,0}, {12,47,39,34,41,52,61,0}, {13,54,49,50,38,33,40,0}, {12,47,58,48,39,34,41,0}, 
  {12,47,33,39,34,60,64,0}, {12,47,52,64,39,40,39,0}, {12,47,35,33,44,33,37,0}, {2,43,56,34,35,35,38,0}, 
  {16,56,35,37,41,40,54,0}, {12,47,40,33,45,34,33,0}, {12,42,34,42,40,35,32,0}, {12,42,38,40,37,61,55,0}, 
  {13,54,33,33,33,36,33,0}, {2,43,39,33,34,34,37,0}, {12,47,46,41,40,38,34,0}, {20,43,42,60,41,41,40,0}, 
  {2,45,41,36,38,37,41,0}, {12,47,40,40,34,41,42,0}, {11,42,50,40,32,35,39,0}, {12,47,37,38,33,48,44,0}, 
  {12,47,38,37,35,47,49,0}, {2,43,37,37,34,41,35,0}, {12,51,50,34,34,41,37,0}, {12,47,38,33,34,41,52,0}, 
  {16,49,58,43,40,32,37,0}, {12,47,48,37,41,33,38,0}, {12,47,38,41,39,50,55,0}, {12,46,59,46,39,40,41,0}, 
  {13,55,34,40,39,35,41,0}, {12,47,48,59,41,33,34,0}, {11,46,49,33,32,37,45,0}, {12,47,33,34,38,35,63,0}, 
  {11,54,62,33,34,35,34,0}, {12,51,58,34,41,36,37,0}, {12,42,40,35,32,54,62,0}, {12,42,35,34,39,64,41,0}, 
  {17,46,41,41,40,32,38,0}, {12,47,37,41,39,58,37,0}, {12,47,46,61,40,34,41,0}, {12,47,35,32,34,37,33,0}, 
  {12,47,37,32,35,60,65,0}, {12,47,52,34,39,34,38,0}, {12,47,41,40,38,65,43,0}, {13,54,33,33,33,36,33,0}, 
  {11,43,40,33,33,40,33,0}, {12,47,43,35,33,34,38,0}, {13,51,32,32,39,33,41,0}, {12,47,35,38,41,57,33,0}, 
  {12,47,54,62,35,39,39,0}, {11,46,38,35,39,60,45,0}, {12,47,48,33,39,37,32,0}, {2,43,57,32,38,38,40,0}, 
  {12,47,40,39,63,37,39,0}, {12,47,40,32,35,41,62,0}, {12,47,46,41,35,35,33,0}, {12,42,32,45,33,32,36,0}, 
  {11,47,51,39,32,37,32,0}, {12,42,34,43,34,37,40,0}, {12,47,48,33,32,37,40,0}, {12,47,58,47,35,39,41,0}, 
  {12,42,55,37,33,39,33,0}, {12,47,39,61,41,32,32,0}, {12,47,40,39,34,57,34,0}, {10,44,37,35,37,33,34,0}, 
  {12,47,39,40,38,53,39,0}, {12,47,40,32,41,39,44,0}, {12,47,35,34,37,38,46,0}, {12,47,48,33,37,37,33,0}, 
  {12,47,54,57,41,38,39,0}, {12,42,32,32,35,51,37,0}, {12,47,50,34,41,37,37,0}, {12,47,33,41,33,41,42,0}, 
  {12,42,35,38,41,37,35,0}, {12,32,33,41,40,39,74,0}, {12,47,34,37,38,51,59,0}, {13,54,40,33,41,39,37,0}, 
  {12,47,41,39,33,48,59,0}, {12,45,41,35,37,39,63,0}, {20,43,34,61,34,38,45,0}, {20,43,47,34,39,34,62,0}, 
  {20,43,45,36,39,38,38,32}, {13,54,40,33,41,39,37,0}, {15,47,33,33,41,38,33,0}, {12,47,32,57,34,38,41,0}
};

char *p;
char *buf;
int plr_index;
int class_array[TOTAL_BLOCK_NUM][8];
int x_array[TOTAL_BLOCK_NUM][10];
int y_array[TOTAL_BLOCK_NUM][2];
char *plr_name;

//将车牌图片嵌入背景图片的指定位置处
void insert_plateImg_into_srcImg(int x, int y, int width, int height, Mat &bg, Mat &plr)
{
  plr.copyTo(bg(Rect(x, y, width, height)));
}

int write_label_xml(const char *filename, int index)
{
  int fd = -1, len, size, ret;
  if ((fd = open(filename, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR)) < 0)
  {
    printf("ERROR, can not open voc xml file: %s\n", filename);
    return 1;
  }
  memset(p, 0x0, TOTAL_BLOCK_NUM * 8 * 200 + 300);
  memset(buf, 0x0, LOOP_BUF_SIZE);
  char *p1 = p;
  sprintf(buf, "%s%d%s%d%s", XML_A, index, XML_B, index, XML_C);
  len = strlen(buf);
  memcpy(p1, buf, len);
  p1 += len;
  int i, j, k;
  for (i = 0; i < TOTAL_BLOCK_NUM; ++i)
  {
    //7位车牌
    k = 0;
    if (0 == licence_chr_class[plr_index][7])
    {
      for (j = 0; j < 7; ++j)
      {
        if (2 == k)
          k++;
        memset(buf, 0x0, LOOP_BUF_SIZE);
        sprintf(buf, "%s%d%s%d%s%d%s%d%s%d%s", XML_LOOP_A, class_array[i][j], XML_LOOP_B, 
               x_array[i][k], XML_LOOP_C, y_array[i][0], XML_LOOP_D, x_array[i][k + 1], XML_LOOP_E, 
               y_array[i][1], XML_LOOP_F);
        len = strlen(buf);
        memcpy(p1, buf, len);
        p1 += len;
        k++;
      }
    }
    //8位车牌
    else
    {
      for (j = 0; j < 8; ++j)
      {
        if (2 == k)
          k++;
        memset(buf, 0x0, LOOP_BUF_SIZE);
        sprintf(buf, "%s%d%s%d%s%d%s%d%s%d%s", XML_LOOP_A, class_array[i][j], XML_LOOP_B, 
               x_array[i][k], XML_LOOP_C, y_array[i][0], XML_LOOP_D, x_array[i][k + 1], XML_LOOP_E, 
               y_array[i][1], XML_LOOP_F);
        len = strlen(buf);
        memcpy(p1, buf, len);
        p1 += len;
        k++;
      }
    }
  }
  memset(buf, 0x0, LOOP_BUF_SIZE);
  sprintf(buf, "%s", XML_D);
  len = strlen(buf);
  memcpy(p1, buf, len);
  p1 += len;
  size = strlen(p);
  if ((ret = write(fd, p, size)) < 0)
  {
    printf("ERROR, write voc xml file failed, write size is incorrect: %s\n", filename);
    close(fd);
    exit(1);
  }
  close(fd);
  return 0;
}

void process_single_background_image(const char *bgFile, const char *plrPath, int block_num, const char *outputFile)
{
  int i, j, k, x, y, z = 0, plr_idx, tmp;
  struct timeval tv;
  Mat bg = imread(bgFile, CV_LOAD_IMAGE_COLOR);
  for (i = 0; i < block_num; ++i)
  {
    for (j = 0; j < block_num; ++j)
    {
      gettimeofday(&tv, NULL);
      srand(tv.tv_sec + tv.tv_usec);
      //在车牌图片库中的索引
      plr_idx = rand() % 100;
      plr_index = plr_idx;
      
    }
  }
}
