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

