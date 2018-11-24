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
  char *model = argv[1];
  char *proto = argv[2];
  Phase phase = TEST;
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
  boost::shared_ptr< Net<float> > net(new caffe::Net<float>(proto, phase));
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(model, &param);
  int num_layers = param.layer_size();
  for (int i = 0; i < num_layers; ++i)
  {
    cout << "Layer " << i << " : " << param.layer(i).name() << "\t" << param.layer(i).type();
    if (param.layer(i).type() == "Convolution")
    {
      ConvolutionParameter conv_param = param.layer(i).convolution_param();
      printf("kernel size : %d\n", conv_param,kernel_size());
    }
  }
  
  return 0;
}
