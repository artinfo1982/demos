#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;

#define BATCH 100
#define B_FILE_NUM 30
#define PIC_SIZE 300
#define PIX_NUM 90000

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    cout << "Usage: " << argv[0] << " img_path batch_path label_file" << endl;
    cout << "Note: any path should end with /" << endl;
    exit(1);
  }
  int i, j = 0, k, m, n;
  int shape[4] = {BATCH, 3, PIC_SIZE, PIC_SIZE};
  float *r = new float[PIX_NUM];
  float *g = new float[PIX_NUM];
  float *b = new float[PIX_NUM];
  float *label = new float[3000];
  float *p = label;
  string line;
  ifstream in_label(argv[3]);
  k = 0;
  if (in_label)
  {
    while (getline(in_label, line))
    {
      label[k] = stof(line, 0);
      k++;
    }
  }
  string s_img_file_name, s_batch_file_name;
  const char* img_file_name;
  const char* batch_file_name;
  int begin;
  int idx = 0;
  Size dsize = Size(PIC_SIZE, PIC_SIZE);
  string prefix = "batch";
  for (i = 0; i < B_FILE_NUM; ++i)
  {
    s_batch_file_name = argv[2] + prefix + to_string(i);
    batch_file_name = s_batch_file_name.c_str();
    FILE* file = fopen(batch_file_name, "w+");
    if (0 == file)
    {
      cout << "batch file is not exist, filename: " << batch_file_name << endl;
      exit(1);
    }
    fwrite(shape, sizeof(int), 4, file);
    begin = i * 100;
    for (j = begin; j < begin + 100; ++j)
    {
      s_img_file_name = argv[1] + to_string(j) + ".jpg";
      img_file_name = s_img_file_name.c_str();
      Mat src = imread(img_file_name);
      Mat img;
      resize(src, img, dsize, 0, 0, INTER_LINEAR);
      memset(r, 0x0, PIX_NUM);
      memset(g, 0x0, PIX_NUM);
      memset(b, 0x0, PIX_NUM);
      k = 0;
      for (m = 0; m < PIC_SIZE; ++m)
      {
        for (n = 0; n < PIC_SIZE; ++n)
        {
          Vec3b pix = img.at<Vec3b>(m, n);
          r[k] = pix[0];
          g[k] = pix[1];
          b[k] = pix[2];
          k++;
        }
      }
      fwrite(r, sizeof(float), PIX_NUM, file);
      fwrite(g, sizeof(float), PIX_NUM, file);
      fwrite(b, sizeof(float), PIX_NUM, file);
      src.release();
      img.release();
    }
    fwrite(p, sizeof(float), PIX_NUM, file);
    p += PIX_NUM;
    if (0 != file)
      fclose(file);
    idx++;
  }
  return 0;
}
