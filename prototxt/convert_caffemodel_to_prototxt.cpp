//g++ test.cpp -o test -I /usr/local/cuda/include -lprotobuf -lboost_system -lcaffe
#include "caffe/caffe.hpp"
#include <string.h>
#include <vector>
#include <stdio.h>

using namespace caffe;
using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Usage: %s caffeModelFileName prototxtFileName\n", argv[0]);
    exit(1);
  }
}
