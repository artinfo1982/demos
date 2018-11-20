#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "PlrChrGen.h"

//省份简称
const char chr_zhs_1[31][4] = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", 
                              "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"};
//省份简称+教练车
const char chr_zhs_2[32][4] = {"京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", 
                              "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "学"};
//军警
const char chr_zhs_3[12][4] = {"警", "军", "海", "空", "南", "济", "沈", "京", "广", "兰", "成", "北"};
//使领馆
const char chr_zhs_4[2][4] = {"使", "领"};
//纯数字
const char chr_eng_1[10][2] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
//英文+数字
const char chr_eng_2[34][2] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", 
                              "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
//纯英文字母
const char chr_eng_3[24][2] = {"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", 
                               "U", "V", "W", "X", "Y", "Z"};

//蓝色像素（小轿车），BGR顺序
CvScalar blue_pixel = CvScalar(132, 20, 8);
//绿色像素（新能源），BGR顺序
CvScalar blue_pixel = CvScalar(123, 242, 49);
//黄色像素1（大型客货车），BGR顺序
CvScalar blue_pixel = CvScalar(0, 168, 216);
//黄色像素2（小型货车），BGR顺序
CvScalar blue_pixel = CvScalar(0, 245, 255);
//白色像素（军警），BGR顺序
CvScalar blue_pixel = CvScalar(255, 255, 255);
//黑色像素（使领馆），BGR顺序
CvScalar blue_pixel = CvScalar(0, 0, 0);

//背景图片
IplImage *bg;
//蓝牌初始框，30x30
IplImage *blue_img_1;
//蓝牌缩放框，25x35
IplImage *blue_img_1;
//普通黄牌初始框，30x30
IplImage *yellow_a_img_1;
//普通黄牌缩放框，25x35
IplImage *yellow_a_img_2;
//小黄牌初始框，30x30
IplImage *yellow_b_img_1;
//小黄牌缩放框，25x35
IplImage *yellow_b_img_2;
//绿牌初始框，30x30
IplImage *green_img_1;
//绿牌缩放框，25x35
IplImage *green_img_2;
//白牌初始框，30x30
IplImage *white_img_1;
//白牌缩放框，25x35
IplImage *white_img_2;
//黑牌初始框，30x30
IplImage *black_img_1;
//黑牌缩放框，25x35
IplImage *black_img_2;

CvxText *zhs_font;
CvxText *eng_font;

char *p;
char *buf;
int total_block_num;
int *class_array;
int *x_array;
int *y_array;

//创建一个画图的矩形框
IplImage *create_plate_image(int width, int height)
{
  return cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
}

//初始化需要的字体
void init_zhs_eng_font(const char *zhs_font_file, const char *eng_font_file)
{
  zhs_font = new CvxText(zhs_font_file);
  eng_font = new CvxText(eng_font_file);
}

//初始化各矩形框内存分配
void init_image_rect_memory()
{
  blue_img_1 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  blue_img_2 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  yellow_a_img_1 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  yellow_a_img_2 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  yellow_b_img_1 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  yellow_b_img_2 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  green_img_1 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  green_img_2 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  white_img_1 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  white_img_2 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  black_img_1 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
  black_img_2 = create_plate_image(RECT_WIDTH_INIT, RECT_HEIGHT_INIT);
}

void release_all()
{
  cvReleaseImage(&blue_img_1);
  cvReleaseImage(&blue_img_2);
  cvReleaseImage(&yellow_a_img_1);
  cvReleaseImage(&yellow_a_img_2);
  cvReleaseImage(&yellow_b_img_1);
  cvReleaseImage(&yellow_b_img_2);
  cvReleaseImage(&green_img_1);
  cvReleaseImage(&green_img_2);
  cvReleaseImage(&white_img_1);
  cvReleaseImage(&white_img_2);
  cvReleaseImage(&black_img_1);
  cvReleaseImage(&black_img_2);
}

//给画图矩形框预设颜色
void set_image_color(IplImage *img, CvScalar &pixel)
{
  int i, j, w, h;
  w = img->width;
  h = img->height;
  for (i = 0; i < h; ++i)
  {
    for (j = 0; j < w; ++j)
      cvSet2D(img, i, j, pixel);
  }
}

//初始化各矩形框的颜色
void init_imae_rect_color()
{
  set_image_color(blue_img_1, blue_pixel);
  set_image_color(yellow_a_img_1, yellow_pixel_1);
  set_image_color(yellow_b_img_1, yellow_pixel_2);
  set_image_color(green_img_1, green_pixel);
  set_image_color(white_img_1, white_pixel);
  set_image_color(black_img_1, black_pixel);
}

//生成高斯噪声
double generate_gaussian_noise(double mu, double sigma)
{
}
