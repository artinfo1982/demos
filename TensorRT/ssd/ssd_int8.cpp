#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <iterator>
#include <algorithm>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"

#include "data_loader.h"
#include "ssd_iplugin.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static const int INPUT_C = 3;
static const int INPUT_H = 300;
static const int INPUT_W = 300;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21; // number of classes
static const int KEEP_TOPK = 200; // number of total bboxes to be kept per image after NMS step

const std::string gCLASSES[OUTPUT_CLS_SIZE]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
    "train", "tvmonitor" };

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME0 = "detection_out";
const char* OUTPUT_BLOB_NAME1 = "keep_count";

struct PPM
{
    int h, w, max;
    uint8_t buffer[INPUT_C * INPUT_H * INPUT_W];
};

struct RES
{
    float totalTime;
    float top1_success;
    float top5_success;
};

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/ssd/", "data/samples/ssd/"};
    return locateFile(input, dirs);
};

class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
  public:
    Int8EntropyCalibrator(DataLoader* dataloader, int batch, int height, int width, int channel, bool readCache = true)
      : mReadCache(readCache)
    {
	    _dataloader = dataloader;
	    DimsNCHW dims = DimsNCHW(batch, channel, height, width);
	    mInputCount1 = batch * dims.c() * dims.h() * dims.w();
	    CHECK(cudaMalloc(&mDeviceInput1, mInputCount1 * sizeof(float)));
	    mInputCount2 = batch * 3;
	    CHECK(cudaMalloc(&mDeviceInput2, mInputCount2 * sizeof(float)));
    }
    virtual ~Int8EntropyCalibrator()
    {
	    CHECK(cudaFree(mDeviceInput1));
	    CHECK(cudaFree(mDeviceInput2));
    }
    int getBatchSize() const override { return 2; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
	    if(!_dataloader->next())
	      return false;
      CHECK(cudaMemcpy(mDeviceInput1, _dataloader->getBatch(),  mInputCount1 * sizeof(float), cudaMemcpyHostToDevice));
	    CHECK(cudaMemcpy(mDeviceInput2, _dataloader->getIminfo(), mInputCount2 * sizeof(float), cudaMemcpyHostToDevice));
      bindings[0] = mDeviceInput1;
      bindings[1] = mDeviceInput2;
      return true;
    }
    const void* readCalibrationCache(size_t& length) override
    {
	    std::cout << "Reading from cache: "<< calibrationTableName()<<std::endl;
	    mCalibrationCache.clear();
	    std::ifstream input(calibrationTableName(), std::ios::binary);
	    input >> std::noskipws;
	    if (mReadCache && input.good())
	      std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
	    length = mCalibrationCache.size();
	    return length ? &mCalibrationCache[0] : nullptr;
    }
    void writeCalibrationCache(const void* cache, size_t length) override
    {
	    std::ofstream output(calibrationTableName(), std::ios::binary);
	    output.write(reinterpret_cast<const char*>(cache), length);
    }
  private:
    static std::string calibrationTableName()
    {
        return std::string("CalibrationTable") + "vgg16";
    }
    bool mReadCache{ true };
    size_t mInputCount1;
    size_t mInputCount2;
    void* mDeviceInput1{ nullptr };
    void* mDeviceInput2{ nullptr };
    std::vector<char> mCalibrationCache;
    DataLoader* _dataloader;
};
