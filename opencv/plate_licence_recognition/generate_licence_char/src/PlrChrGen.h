#ifndef PLR_GEN_H
#define PLR_GEN_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <stdio.h>
#include <ftglyph.h>
#include <sys/time.h>
#include <limits>
#include <cmath>
#include "CvxText.h"

//文字矩形框初始宽度
#define RECT_WIDTH_INIT 30
//文字矩形框初始高度
#define RECT_HEIGHT_INIT 30
//文字矩形框缩放后宽度
#define RECT_WIDTH_SCALE 25
//文字矩形框缩放后高度
#define RECT_HEIGHT_SCALE 35
//背景图片的边长，默认为正方形
#define SRC_IMAGE_SIDE_LEN 512
//将原始的背景图片划分为若干个小正方形，每个的边长，在每一个小正方形中随机生成文字图片
#define DIV_SIDE_LEN 64
//方块内坐标游走的最大范围
#define COOR_RANGE 30
//汉字出现在车牌小框内的X起始坐标
#define ZHS_X_BEGIN 7
//汉字出现在车牌小框内的Y起始坐标
#define ZHS_Y_BEGIN 23
//英文或数字出现在车牌小框内的X起始坐标
#define ENG_X_BEGIN 12
//英文或数字出现在车牌小框内的Y起始坐标
#define ENG_Y_BEGIN 22
//汉字字体大小
#define ZHS_FONT_SIZE 18
//英文或数字字体大小
#define ENG_FONT_SIZE 22

//拼接xml示例澹?
//XML_A + 文件名编号 + XML_B + 文件名编号 + XML_C + [ XML_LOOP_A + 类别名 + XML_LOOP_B + xmin
//+ XML_LOOP_C + ymin + XML_LOOP_D + xmax + XML_LOOP_E + ymax + XML_LOOP_F ] + XML_D
#define XML_A "<annotation><folder>plr</folder><filename>"
#define XML_B "</filename><path>/home/cd/plr/VOCdevkit/VOC2007/JPEGImages/"
#define XML_C ".jpg</path><source><database>Unknown</database></source><size><width>512</width><height>512</height><depth>3</depth></size><segmented>0</segmented>"
#define XML_LOOP_A "<object><name>"
#define XML_LOOP_B "</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult><bndbox><xmin>"
#define XML_LOOP_C "</xmin><ymin>"
#define XML_LOOP_D "</ymin><xmax>"
#define XML_LOOP_E "</xmax><ymax>"
#define XML_LOOP_F "</ymax></bndbox></object>"
#define XML_D "</annotation>"

IplImage *create_plate_image(int width, int height);
void init_zhs_eng_font(const char *zhs_font_file, const char *eng_font_file);
void init_image_rect_memory();
void release_all();
void set_image_color(IplImage *img, CvScalar &pixel);
void init_image_rect_color();
double generate_gaussian_noise(double mu, double sigma);
void add_gauss_noise(IplImage *src);
void add_warp_affine(IplImage *src, CvScalar &scalar);
void put_char_into_rect_and_resize(CvScalar &scalar, CvxText *font, int size, IplImage *src, IplImage *dst, const char *chr, int x, int y, int red, int green, int blue);
void insert_plateImg_into_srcImg(int x, int y, int width, int height, IplImage *src, IplImage *plr);
int write_label_xml(const char *filename, int index);
void process_single_background_image(const char *inputFile, int block_num, const char *outputFile);
int class_parser(const char *class_str);

extern char *p;
extern char *buf;
extern int total_block_num;
extern int *class_array;
extern int *x_array;
extern int *y_array;

#endif
