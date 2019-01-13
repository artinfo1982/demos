#ifndef __SSD_IPLUGIN__
#define __SSD_IPLUGIN__

#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <iterator>
#include <algorithm>
#include <vector>
#include <memory>
#include <iostream>

#include "common.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

void cudaSoftmax(int n, int channels, float *x, float *y);

class SoftmaxPlugin : public IPlugin
{
public:
  SoftmaxPlugin() {}
  SoftmaxPlugin(const void* buffer, size_t size)
  {
    assert(size == sizeof(mCopySize));
    mCopySize = *reinterpret_cast<const size_t*>(buffer);
  }
  int initialize() override { return 0; }
}

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
  virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;
  IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
  void(*nvPluginDeleter)(INvPlugin*)
  {
    [](INvPlugin* ptr)
    {
      ptr->destroy();
    }
  };
  bool isPlugin(const char* name) override;
  void destroyPlugin();
  
  //normalize layer
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mNormalizeLayer{ nullptr, nvPluginDeleter };
  //priorbox layers
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv4_3_norm_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mFc7_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv6_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv7_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv8_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv9_2_mbox_priorbox_layer{ nullptr, nvPluginDeleter };
  //detection output layer
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mDetection_out{ nullptr, nvPluginDeleter };
  //permute layer
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv4_3_norm_mbox_loc_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv4_3_norm_mbox_conf_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mFc7_mbox_loc_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mFc7_mbox_conf_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv6_2_mbox_loc_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv6_2_mbox_conf_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv7_2_mbox_loc_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv7_2_mbox_conf_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv8_2_mbox_loc_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv8_2_mbox_conf_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv9_2_mbox_loc_permute_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mConv9_2_mbox_conf_permute_layer{ nullptr, nvPluginDeleter };
  //concat layer
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mMbox_loc_concat_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mMbox_conf_concat_layer{ nullptr, nvPluginDeleter };
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mMbox_priorbox_concat_layer{ nullptr, nvPluginDeleter };
};

#endif
