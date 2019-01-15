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

template <int OutC>
class Reshape : public IPlugin
{
  public:
    Reshape() {}
    Reshape(const void *buffer, size_t size)
    {
      assert(size == sizeof(mCopySize));
      mCopySize = *reinterpret_cast<const size_t *>(buffer);
    }
    int getNbOutputs() const override { return 1; }
    dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
      assert(1 == nbInputDims);
      assert(0 == index);
      assert(3 == inputs[index].nbDims);
      assert((inputs[0].d[0]) * (inputs[0].d[1]) % OutC == 0);
      return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
    }
    int initialize() override { return 0; }
    void terminate() override {}
    size_t getWorkspaceSize(int) const override { return mCopySize * 1; }
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override
    {
      CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
      return 0;
    }
    size_t getSerializationSize() override { return sizeof(mCopySize); }
    void serialize(void *buffer) override { *reinterpret_cast<size_t *>(buffer) = mCopySize; }
    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override
    {
      mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }
  
  protected:
    size_t mCopySize;
};

class Flatten : public IPlugin
{
  public:
    Flatten() {}
    Flatten(const void *buffer, size_t size)
    {
      assert(size == 3 * sizeof(int));
      const int *d = *reinterpret_cast<const int *>(buffer);
      _size = d[0] * d[1] * d[2];
      dimBottom = DimsCHW{d[0], d[1], d[2]};
    }
    inline int getNbOutputs() const override { return 1; }
    dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override
    {
      assert(1 == nbInputDims);
      assert(0 == index);
      assert(3 == inputs[index].nbDims);
      _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
      return DimsCHW(_size, 1, 1);
    }
    int initialize() override { return 0; }
    void terminate() override {}
    size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void *const *inputs, void **outputs, void *, cudaStream_t stream) override
    {
      CHECK(cudaMemcpyAsync(outputs[0], inputs[0], batchSize * _size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      return 0;
    }
    size_t getSerializationSize() override { return 3 * sizeof(int); }
    void serialize(void *buffer) override
    {
      int *d = reinterpret_cast<int *>(buffer);
      d[0] = dimBottom.c();
      d[1] = dimBottom.h();
      d[2] = dimBottom.w();
    }
    void configure(const Dims *inputs, int nbInputs, const Dims *outputs, int nbOutputs, int) override
    {
      dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }
    
  protected:
    DimsCHW dimBottom;
    int _size;
}

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
  inline void terminate() override {}
  inline int getNbOutputs() const override { return 1; }
  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
  {
    assert(nbInputDims == 1);
    assert(index == 0);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
  }
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
