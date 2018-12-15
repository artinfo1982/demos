//g++ convert_caffemodel_to_prototxt.cpp -o test -O3 -I /usr/local/cuda/include -lprotobuf -lcaffe -lboost_system

#include "caffe/caffe.hpp"
#include <string.h>
#include <vector>
#include <stdio.h>
#include <fstream>

using namespace caffe;
using namespace std;

string filler_parameter_parser(FillerParameter &fp)
{
  string s;
  if (fp.has_type())
    s.append("      type: \"").append(fp.type()).append("\"\n");
  if (fp.has_value())
    s.append("      value: \"").append(to_string(fp.value())).append("\n");
  if (fp.has_min())
    s.append("      min: \"").append(to_string(fp.min())).append("\n");
  if (fp.has_max())
    s.append("      max: \"").append(to_string(fp.max())).append("\n");
  if (fp.has_mean())
    s.append("      mean: \"").append(to_string(fp.mean())).append("\n");
  if (fp.has_std())
    s.append("      std: \"").append(to_string(fp.std())).append("\n");
  if (fp.has_sparse())
    s.append("      sparse: \"").append(to_string(fp.sparse())).append("\n");
  return s;
}

string input_parameter_parser(LayerParameter &lp)
{
  string s;
  InputParameter ip = lp.input_param();
  if (ip.shape_size() > 0)
  {
    s.append("  input_param {\n");
    for (int i=0; i<ip.shape_size(); ++i)
    {
      BlobShape bs = ip.shape(i);
      s.append("    shape: {\n");
      if (bs.dim_size() > 0)
      {
        for (int j=0; j<bs.dim_size(); ++j)
          s.append("      dim: ").append(to_string(bs.dim(j))).append("\n");
      }
    }
    s.append("    }\n  }\n");
    return s;
  }
  else
    return "";
}

string data_parameter_parser(LayerParameter &lp)
{
  string s;
  DataParameter dp = lp.data_param();
  s.append("  data_param {\n");
  if (dp.has_source())
    s.append("    source: \"").append(dp.source()).append("\"\n");
  if (dp.has_mean_file())
    s.append("    mean_file: \"").append(dp.mean_file()).append("\"\n");
  if (dp.has_batch_size())
    s.append("    batch_size: ").append(to_string(dp.batch_size())).append("\n");
  if (dp.has_crop_size())
    s.append("    crop_size: ").append(to_string(dp.crop_size())).append("\n");
  if (dp.has_mirror())
    s.append("    mirror: ").append(to_string(dp.mirror())).append("\n");
  if (dp.has_rand_skip())
    s.append("    rand_skip: ").append(to_string(dp.rand_skip())).append("\n");
  if (dp.has_scale())
    s.append("    scale: ").append(to_string(dp.scale())).append("\n");
  if (dp.has_prefetch())
    s.append("    prefetch: ").append(to_string(dp.prefetch())).append("\n");
  s.append("  }\n");
}

string convolution_parameter_parser(LayerParameter &lp)
{
  string s;
  ConvolutionParameter cp = lp.convolution_param();
  s.append("  convolution_param {\n");
  if (cp.has_num_output())
    s.append("    num_output: ").append(to_string(cp.num_output())).append("\n");
  if (cp.kernel_size_size() > 0)
    s.append("    kernel_size: ").append(to_string(cp.kernel_size_size())).append("\n");
  if (cp.stride_size() > 0)
    s.append("    stride_size: ").append(to_string(cp.stride_size())).append("\n");
  if (cp.pad_size() > 0)
    s.append("    pad_size: ").append(to_string(cp.pad_size())).append("\n");
  if (cp.has_weight_filler())
  {
    s.append("    weight_filler {\n");
    FillerParameter fp = cp.weight_filler();
    s.append(filler_parameter_parser(fp)).append("    }\n")
  }
  if (cp.has_bias_filler())
  {
    s.append("    bias_filler {\n");
    FillerParameter fp = cp.bias_filler();
    s.append(filler_parameter_parser(fp)).append("    }\n")
  }
    
}



