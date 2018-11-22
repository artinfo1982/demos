#ifndef PLR_INTO_SRC_H
#define PLR_INTO_SRC_H

#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <sys/time.h>
#include <limits.h>
#include <cmath>

using namespace cv;

//背景图片的边长，默认为正方形
#define SRC_IMAGE_SIDE_LEN 512
//将原始的背景图片划分为若干个小正方形，每个的边长，在每一个小正方形中随机插入车牌图片
#define DIV_SIDE_LEN 128
//总共有多少个小正方形划分的区域，(SRC_IMAGE_SIDE_LEN/DIV_SIDE_LEN)^2
#define TOTAL_BLOCK_NUM 16
//方块内X坐标游走的最大范围
#define COOR_RANGE_X 30
//方块内Y坐标游走的最大范围
#define COOR_RANGE_Y 100
//背景图片的个数
#define MAX_BG_FILE_NUM 50
//车牌图片的个数
#define MAX_PLR_FILE_NUM 100
//xml物体信息循环体缓存大小
#define LOOP_BUF_SIZE 1024

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

void insert_plateImg_into_srcImg(int x, int y, int width, int height, Mat &bg, Mat &plr);
int write_label_xml(const char *filename, int index);
void process_single_background_image(const char *bgFile, const char *plrPath, int block_num, const char*outputFile);

extern char *p;
extern char *buf;
extern int plr_index;
extern char *plr_name;
extern int class_array[TOTAL_BLOCK_NUM][8];
extern int x_array[TOTAL_BLOCK_NUM][10];
extern int y_array[TOTAL_BLOCK_NUM][2];

#endif
