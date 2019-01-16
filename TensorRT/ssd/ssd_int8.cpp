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
	mInputCount = batch * dims.c() * dims.h() * dims.w();
	CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    }
    virtual ~Int8EntropyCalibrator()
    {
	CHECK(cudaFree(mDeviceInput));
    }
    int getBatchSize() const override { return 2; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
	if(!_dataloader->next())
	    return false;
      	CHECK(cudaMemcpy(mDeviceInput, _dataloader->getBatch(),  mInputCount * sizeof(float), cudaMemcpyHostToDevice));
      	bindings[0] = mDeviceInput;
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
        return std::string("CalibrationTable_SSD_VGG16");
    }
    bool mReadCache{ true };
    size_t mInputCount;
    void* mDeviceInput{ nullptr };
    std::vector<char> mCalibrationCache;
    DataLoader* _dataloader;
};

void readPPMFile(const std::string& filename, PPM& ppm)
{
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    infile.seekg(3, infile.beg);
    infile >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
    infile.close();
}

void caffeToTRTModel(const std::string& deployFile, const std::string& modelFile, const std::vector<std::string>& outputs, 
		     unsigned int maxBatchSize, nvcaffeparser1::IPluginFactory* pluginFactory, IHostMemory **modelStream, DataType dataType)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);
    std::cout << "Begin to parse model" <<std::endl;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, 
							      dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);
    std::cout << "End to parse model" << std::endl;
    for (auto& s : outputs)
	network->markOutput(*blobNameToTensor->find(s.c_str()));
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setInt8Mode(true);
    DataLoader* dataLoader = new DataLoader(maxBatchSize, "/home/cd/TensorRT-4.0.1.6/data/ssd/list.txt", 300, 300, 3);
    Int8EntropyCalibrator* calibrator = new Int8EntropyCalibrator(dataLoader, maxBatchSize, 300, 300, 3);
    builder->setInt8Calibrator(calibrator);
    std::cout << "Begin to build engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);
    std::cout << "End to build engine..." << std::endl;
    network->destroy();
    parser->destroy();
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

float doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 3);
    void* buffers[3];
    float ms = 0.0f;
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
	outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
	outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
    CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * KEEP_TOPK * 7 * sizeof(float)));                  // detection_out
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * sizeof(int))); // keepCount
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), 
	cudaMemcpyHostToDevice, stream));
    cudaStreamSynchronize(stream);
    double start = std::clock();
    int iter = 1;
    for(int i=0; i<iter; ++i)
    {
        context.enqueue(batchSize, buffers, stream, nullptr);
        cudaStreamSynchronize(stream);
    }
    ms = (std::clock()-start) / (double) CLOCKS_PER_SEC /iter * 1000;
    std::cout<< "infer total time elapse:  "<< ms << " ms" <<std::endl;
    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * KEEP_TOPK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
    cudaStreamDestroy(stream);
    return ms;
}
