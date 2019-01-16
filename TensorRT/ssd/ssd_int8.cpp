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
}

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

bool cmp(const std::pair<std::string, float> a, const std::pair<std::string, float> b)
{
    return a.second > b.second;
}

RES do_each_batch(unsigned int N, int beginIdx, IExecutionContext *context, const std::vector<std::string> classList, 
		 float *data, float *detectionOut, PPM *ppms, int* keepCount)
{
    memset(data, 0x0, N * INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    memset(detectionOut, 0x0, N * KEEP_TOPK * 7 * sizeof(float));
    memset(ppms, 0x0, N * sizeof(PPM));
    memset(keepCount, 0x0, N * sizeof(int));
    PluginFactory pluginFactory;
    RES res = {0};
    std::vector<std::string> imageList;
    std::vector<std::string> clazList;
    for (unsigned int i = beginIdx; i < beginIdx + N; ++i)
    {
	imageList.push_back("/home/cd/TensorRT-4.0.1.6/data/ssd/ppms/" + std::to_string(i) + ".ppm");
	clazList.push_back(classList[i]);
    }
    assert(imageList.size() == N);
    assert(clazList.size() == N);
    for (unsigned int i = 0; i < N; ++i)
    	readPPMFile(imageList[i], ppms[i]);
    // pixel mean used by the Faster R-CNN's author
    float pixelMean[3]{ 104.0f, 117.0f, 123.0f }; // also in BGR order
    for (unsigned int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
    {
	for (unsigned int c = 0; c < INPUT_C; ++c)
	{
	    // the color image to input should be in BGR order
	    for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
		data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
	}
    }
    res.totalTime = doInference(*context, data, detectionOut, keepCount, N);
    int t1_success = 0, t5_success = 0;
    float class_max_prob = 0.0f;
    for (unsigned int p = 0; p < N; ++p)
    {
	std::vector<std::pair<std::string, float>> vec;
	for (int i = 1; i < keepCount[p]; ++i)
	{
	    class_max_prob = 0.0f;
	    float *det = detectionOut + (p * KEEP_TOPK + i) * 7;
	    assert((int)det[1] < OUTPUT_CLS_SIZE);
	    class_max_prob = MAX(class_max_prob, det[2]);
	    vec.push_back(make_pair(gCLASSES[(int)det[1]].c_str(), class_max_prob));
	}
	sort(vec.begin(), vec.end(), cmp);
	if (vec[0].first == clazList[p])
	    t1_success++;
	if ((vec[0].first == clazList[p]) || (vec[1].first == clazList[p]) || (vec[2].first == clazList[p]) || 
	    (vec[3].first == clazList[p]) || (vec[4].first == clazList[p]))
	    t5_success++;
	std::vector<std::pair<std::string, float>>().swap(vec);
    }
    res.top1_success = t1_success;
    res.top5_success = t5_success;
    std::vector<std::string>().swap(imageList);
    std::vector<std::string>().swap(clazList);
    return res;
}

int main(int argc, char* argv[])
{
    PluginFactory pluginFactory;
    IHostMemory *modelStream{ nullptr };
    const int N = 10;
    const int total_number = 3000;
    caffeToTRTModel("/home/cd/TensorRT-4.0.1.6/data/ssd/ssd_iplugin.prototxt", 
		    "/home/cd/TensorRT-4.0.1.6/data/ssd/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel", 
		    std::vector < std::string > { OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1 }, 
		    N, &pluginFactory, &modelStream, DataType::kINT8);
    pluginFactory.destroyPlugin();
    std::vector<std::string> classList;
    std::ifstream class_infile("/home/cd/TensorRT-4.0.1.6/data/ssd/classes.txt");
    std::string tmp;
    while (class_infile >> tmp)
	classList.push_back(tmp);
    RES res = {0};
    float totalTime = 0.0f;
    int batch_total_number = 0;
    int top1_success = 0, top5_success = 0;
    float top1_error_rate = 0.0f, top5_error_rate = 0.0f;
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), &pluginFactory);
    IExecutionContext *context = engine->createExecutionContext();
    //save int8 tensorRT model
    std::ofstream outfile("/home/cd/TensorRT-4.0.1.6/data/ssd/ssd-int8.trt", std::ios::out | std::ios::binary);
    if (!outfile.is_open())
    {
	std::cout << "fail to open trt model file" << std::endl;
	exit(1);
    }
    unsigned char* p = (unsigned char*)modelStream->data();
    outfile.write((char*)p, modelStream->size());
    outfile.close();
    float* data = new float[N  *INPUT_C * INPUT_H * INPUT_W];
    float* detectionOut = new float[N * KEEP_TOPK * 7];
    PPM* ppms = new PPM[N];
    int* keepCount = new int[N];
	
    for (int i = 0; i < total_number/N; ++i)
    {
	std::cout << "do batch " << i < " ..." << std::endl;
	res = do_each_batch(N, i * N, context, classList, data, detectionOut, ppms, keepCount);
	totalTime += res.totalTime;
	top1_success += res.top1_success;
	top5_success += res.top5_success;
	batch_total_number += N;
	top1_error_rate = (batch_total_number - top1_success) / (float)batch_total_number * 100.0f;
	top5_error_rate = (batch_total_number - top5_success) / (float)batch_total_number * 100.0f;
	std::cout << "in this batch, avg infer time of each image=" << totalTime / batch_total_number << "ms, "
		<< "top1 error rate=" << top1_error_rate << "%, top5 error rate=" << top5_error_rate << "%"
		<< ", total number=" << batch_total_number << ", total top1_success=" << top1_success 
		<< ", total top5_success=" << top5_success << std::endl;
    }
    return 0;
}
