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

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

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
  
  std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};

#endif
