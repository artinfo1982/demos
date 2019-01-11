#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <iterator>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"
#include <cstdio>
#include <ctime>

#include "data_loader.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static const int INPUT_C = 3;
static const int INPUT_H = 375;
static const int INPUT_W = 500;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const std::string CLASSES[OUTPUT_CLS_SIZE]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", 
					   "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
					   "train", "tvmonitor" };

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";

const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };

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
    std::vector<std::string> dirs{"data/faster-rcnn/", "data/samples/faster-rcnn/"};
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
    DataLoader* dataLoader = new DataLoader(maxBatchSize, "/home/cd/TensorRT-4.0.1.6/data/faster-rcnn/list.txt", 500, 375, 3);
    Int8EntropyCalibrator* calibrator = new Int8EntropyCalibrator(dataLoader, maxBatchSize, 375, 500, 3);
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

float doInference(IExecutionContext& context, float* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, 
		  float *outputRois, int batchSize)
{
    std::cout << "Begin to do infer..." << std::endl;
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 5);
    void* buffers[5];
    float ms = 0.0f;
    int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
	inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1),
	outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
	outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
	outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
    CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
    CHECK(cudaMalloc(&buffers[inputIndex1], batchSize * IM_INFO_SIZE * sizeof(float)));                  // im_info
    CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
    CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), 
			  cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
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
    CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), 
			  cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), 
			  cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    CHECK(cudaFree(buffers[inputIndex0]));
    CHECK(cudaFree(buffers[inputIndex1]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
    cudaStreamDestroy(stream);
    std::cout << "End to do infer..." << std::endl;
    return ms;
}

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
    {
	assert(isPlugin(layerName));
	if (!strcmp(layerName, "RPROIFused"))
	{
	    assert(mPluginRPROI == nullptr);
	    assert(nbWeights == 0 && weights == nullptr);
	    mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
		(createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
		    DimsHW(poolingH, poolingW), Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount },
		    Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount }), nvPluginDeleter);
	    return mPluginRPROI.get();
	}
	else
	{
	    assert(0);
	    return nullptr;
	}
    }
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
	assert(isPlugin(layerName));
	if (!strcmp(layerName, "RPROIFused"))
	{
	    assert(mPluginRPROI == nullptr);
	    mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
	        (createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
	    return mPluginRPROI.get();
	}
	else
	{
	    assert(0);
	    return nullptr;
	}
    }
    bool isPlugin(const char* name) override
    {
	return (!strcmp(name, "RPROIFused"));
    }
    void destroyPlugin()
    {
	mPluginRPROI.release();
	mPluginRPROI = nullptr;
    }
    void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};

void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo, const int N, const int nmsMaxOut, const int numCls)
{
    float width, height, ctr_x, ctr_y;
    float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
    float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
    for (int i = 0; i < N * nmsMaxOut; ++i)
    {
	width = rois[i * 4 + 2] - rois[i * 4] + 1;
	height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
	ctr_x = rois[i * 4] + 0.5f * width;
	ctr_y = rois[i * 4 + 1] + 0.5f * height;
	deltas_offset = deltas + i * numCls * 4;
	predBBoxes_offset = predBBoxes + i * numCls * 4;
	imInfo_offset = imInfo + i / nmsMaxOut * 3;
	for (int j = 0; j < numCls; ++j)
	{
	    dx = deltas_offset[j * 4];
	    dy = deltas_offset[j * 4 + 1];
	    dw = deltas_offset[j * 4 + 2];
	    dh = deltas_offset[j * 4 + 3];
	    pred_ctr_x = dx * width + ctr_x;
	    pred_ctr_y = dy * height + ctr_y;
	    pred_w = exp(dw) * width;
	    pred_h = exp(dh) * height;
	    predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
	    predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
	    predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
	    predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
	}
    }
}

bool cmp(const std::pair<std::string, float> a, const std::pair<std::string, float> b)
{
    return a.second > b.second;
}

RES do_each_batch(unsigned int N, int beginIdx, IExecutionContext *context, const std::vector<std::string> classList, 
		 float *data, float *imInfo, PPM *ppms, float *rois, float *bboxPreds, float *clsProbs, float *preBBoxes)
{
    memset(data, 0x0, N * INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    memset(imInfo, 0x0, N * 3 * sizeof(float));
    memset(ppms, 0x0, N * sizeof(PPM));
    memset(rois, 0x0, N * nmsMaxOut * 4 * sizeof(float));
    memset(bboxPreds, 0x0, N * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float));
    memset(clsProbs, 0x0, N * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float));
    memset(preBBoxes, 0x0, N * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float));
    PluginFactory pluginFactory;
    RES res = {0};
    std::vector<std::string> imageList;
    std::vector<std::string> clazList;
    for (unsigned int i = beginIdx; i < beginIdx + N; ++i)
    {
	imageList.push_back("/home/cd/TensorRT-4.0.1.6/data/faster-rcnn/ppms/" + std::to_string(i) + ".ppm");
	clazList.push_back(classList[i]);
    }
    assert(imageList.size() == N);
    assert(clazList.size() == N);
    for (unsigned int i = 0; i < N; ++i)
    {
	readPPMFile(imageList[i], ppms[i]);
	imInfo[i * 3] = float(ppms[i].h);   // number of rows
	imInfo[i * 3 + 1] = float(ppms[i].w); // number of columns
	imInfo[i * 3 + 2] = 1;         // image scale
    }
    // pixel mean used by the Faster R-CNN's author
    float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f }; // also in BGR order
    for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
    {
	for (int c = 0; c < INPUT_C; ++c)
	{
	    // the color image to input should be in BGR order
	    for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
		data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
	}
    }
    res.totalTime = doInference(*context, data, imInfo, bboxPreds, clsProbs, rois, N);
    for (unsigned int i = 0; i < N; ++i)
    {
	float * rois_offset = rois + i * nmsMaxOut * 4;
	for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
	    rois_offset[j] /= imInfo[i * 3 + 2];
    }
    bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);
    int t1_success = 0, t5_success = 0;
    float class_max_prob = 0.0f;
    for (unsigned int i = 0; i < N; ++i)
    {
	float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
	std::vector<std::pair<std::string, float>> vec;
	for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
	{
	    class_max_prob = 0.0f;
	    for (int r = 0; r < nmsMaxOut; ++r)
	    	class_max_prob = MAX(class_max_prob, scores[r * OUTPUT_CLS_SIZE + c]);
	    vec.push_back(make_pair(CLASSES[c], class_max_prob));
	}
	sort(vec.begin(), vec.end(), cmp);
	if (vec[0].first == clazList[i])
	    t1_success++;
	if ((vec[0].first == clazList[i]) || (vec[1].first == clazList[i]) || (vec[2].first == clazList[i]) || 
	    (vec[3].first == clazList[i]) || (vec[4].first == clazList[i]))
	    t5_success++;
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
    caffeToTRTModel("/home/cd/TensorRT-4.0.1.6/data/faster-rcnn/faster_rcnn_test_iplugin.prototxt", 
		    "/home/cd/TensorRT-4.0.1.6/data/faster-rcnn/VGG16_faster_rcnn_final.caffemodel", 
		    std::vector < std::string > { OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2 }, 
		    N, &pluginFactory, &modelStream, DataType::kINT8);
    pluginFactory.destroyPlugin();
    std::vector<std::string> classList;
    std::ifstream class_infile("/home/cd/TensorRT-4.0.1.6/data/faster-rcnn/classes.txt");
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
    std::ofstream outfile("/home/cd/TensorRT-4.0.1.6/data/faster-rcnn/faster-rcnn-int8.trt", std::ios::out | std::ios::binary);
    if (!outfile.is_open())
    {
	std::cout << "fail to open trt model file" << std::endl;
	exit(1);
    }
    unsigned char* p = (unsigned char*)modelStream->data();
    outfile.write((char*)p, modelStream->size());
    outfile.close();
    float* data = new float[N  *INPUT_C * INPUT_H * INPUT_W];
    float* imInfo = new float[N * 3];
    PPM* ppms = new PPM[N];
    float* rois = new float[N * nmsMaxOut * 4];
    float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
    float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];
    float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
	
    for (int i = 0; i < total_number/N; ++i)
    {
	std::cout << "do batch " << i < " ..." << std::endl;
	res = do_each_batch(N, i * N, context, classList, data, imInfo, ppms, rois, bboxPreds, clsProbs, predBBoxes);
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
