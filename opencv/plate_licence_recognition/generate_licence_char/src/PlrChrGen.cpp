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

