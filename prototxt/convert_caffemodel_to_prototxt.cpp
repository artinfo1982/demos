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
}
