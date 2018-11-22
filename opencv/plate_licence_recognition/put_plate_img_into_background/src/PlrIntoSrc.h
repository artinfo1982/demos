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
