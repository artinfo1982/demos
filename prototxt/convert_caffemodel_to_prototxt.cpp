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
    s.append("      value: ").append(to_string(fp.value())).append("\n");
  if (fp.has_min())
    s.append("      min: ").append(to_string(fp.min())).append("\n");
  if (fp.has_max())
    s.append("      max: ").append(to_string(fp.max())).append("\n");
  if (fp.has_mean())
    s.append("      mean: ").append(to_string(fp.mean())).append("\n");
  if (fp.has_std())
    s.append("      std: ").append(to_string(fp.std())).append("\n");
  if (fp.has_sparse())
    s.append("      sparse: ").append(to_string(fp.sparse())).append("\n");
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
  return s;
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
    s.append(filler_parameter_parser(fp)).append("    }\n");
  }
  if (cp.has_bias_filler())
  {
    s.append("    bias_filler {\n");
    FillerParameter fp = cp.bias_filler();
    s.append(filler_parameter_parser(fp)).append("    }\n");
  }
  s.append("  }\n");
  return s;
}

string lrn_parameter_parser(LayerParameter &lp)
{
  string s;
  LRNParameter rp = lp.lrn_param();
  s.append("  lrn_param {\n");
  if (rp.has_local_size())
    s.append("    local_size: ").append(to_string(rp.local_size())).append("\n");
  if (rp.has_alpha())
    s.append("    alpha: ").append(to_string(rp.alpha())).append("\n");
  if (rp.has_beta())
    s.append("    beta: ").append(to_string(rp.beta())).append("\n");
  if (rp.has_k())
    s.append("    k: ").append(to_string(rp.k())).append("\n");
  s.append("  }\n");
  return s;
}

string pooling_parameter_parser(LayerParameter &lp)
{
  string s;
  PoolingParameter pp = lp.pooling_param();
  s.append("  pooling_param {\n");
  if (pp.has_pool())
    s.append("    pool: ").append(to_string(pp.pool())).append("\n");
  if (pp.has_kernel_size())
    s.append("    kernel_size: ").append(to_string(pp.kernel_size())).append("\n");
  if (pp.has_stride())
    s.append("    stride: ").append(to_string(pp.stride())).append("\n");
  if (pp.has_pad())
    s.append("    pad: ").append(to_string(pp.pad())).append("\n");
  s.append("  }\n");
  return s;
}

string inner_product_parameter_parser(LayerParameter &lp)
{
  string s;
  InnerProductParameter ipp = lp.inner_product_param();
  s.append("  inner_product_param {\n");
  if (ipp.has_num_output())
    s.append("    num_output: ").append(to_string(ipp.num_output())).append("\n");
  if (ipp.has_weight_filler())
  {
    s.append("    weight_filler {\n");
    FillerParameter fp = ipp.weight_filler();
    s.append(filler_parameter_parser(fp)).append("    }\n");
  }
  if (ipp.has_bias_filler())
  {
    s.append("    bias_filler {\n");
    FillerParameter fp = ipp.bias_filler();
    s.append(filler_parameter_parser(fp)).append("    }\n");
  }
  if (ipp.has_axis())
    s.append("    axis: ").append(to_string(ipp.axis())).append("\n");
  s.append("  }\n");
  return s;
}

string dropout_parameter_parser(LayerParameter &lp)
{
  string s;
  DropoutParameter dp = lp.dropout_param();
  s.append("  dropout_param {\n");
  if (dp.has_dropout_ratio())
    s.append("    dropout_ratio: ").append(to_string(dp.dropout_ratio())).append("\n");
  s.append("  }\n");
  return s;
}

void write_prototxt_file(const char *fileName, string &s)
{
  ofstream out(fileName, ios::out | ios::app);
  if (out.is_open())
  {
    out << s;
    out.close();
  }
  else
  {
    printf("ERROR, output prototxt file can not open, fileName: %s\n", fileName);
    exit(1);
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Usage: %s caffeModelFileName prototxtFileName\n", argv[0]);
    exit(1);
  }
  char *model = argv[1];
  char *proto = argv[2];
  string s;
  NetParameter net;
  ReadNetParamsFromBinaryFileOrDie(model, &net);
  int i, j, k;
  if (net.has_name())
    s.append("name: \"").append(net.name()).append("\"\n");
  for (i = 0; i < net.layer_size(); ++i)
  {
    LayerParameter lp = net.layer(i);
    s.append("layer {\n  name: \"").append(lp.name()).append("\"\n  type: \"").append(lp.type()).append("\"\n");
    if (lp.bottom_size() > 0)
    {
      for (j = 0; j < lp.bottom_size(); ++j)
        s.append("  bottom: \"").append(lp.bottom(j)).append("\"\n");
    }
    if (lp.top_size() > 0)
    {
      for (j = 0; j < lp.top_size(); ++j)
        s.append("  top: \"").append(lp.top(j)).append("\"\n");
    }
    if (lp.param_size() > 0)
    {
      for (j = 0; j < lp.param_size(); ++j)
      {
        ParamSpec ps = lp.param(j);
        s.append("  param {\n");
        if (ps.has_lr_mult())
          s.append("    lr_mult: ").append(to_string(ps.lr_mult())).append("\n");
        if (ps.has_decay_mult())
          s.append("    decay_mult: ").append(to_string(ps.decay_mult())).append("\n");
        s.append("  }\n");
      }
    }
    if (lp.type() == "Input")
      s += input_parameter_parser(lp);
    else if (lp.type() == "Data")
      s += data_parameter_parser(lp);
    else if (lp.type() == "Convolution")
      s += convolution_parameter_parser(lp);
    else if (lp.type() == "LRN")
      s += lrn_parameter_parser(lp);
    else if (lp.type() == "Pooling")
      s += pooling_parameter_parser(lp);
    else if (lp.type() == "InnerProduct")
      s += inner_product_parameter_parser(lp);
    else if (lp.type() == "Dropout")
      s += dropout_parameter_parser(lp);
    s.append("}\n");
  }
  write_prototxt_file(proto, s);
  return 0;
}
