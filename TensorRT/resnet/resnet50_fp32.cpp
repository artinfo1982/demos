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

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_CLS_SIZE = 1000; // number of classes

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
    std::vector<std::string> dirs{"data/resnet/", "data/samples/resnet/"};
    return locateFile(input, dirs);
}

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

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
		     unsigned int maxBatchSize, IHostMemory **modelStream, DataType dataType)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    std::cout << "Begin to parse model" <<std::endl;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, 
							      dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);
    std::cout << "End to parse model" << std::endl;
    for (auto& s : outputs)
	network->markOutput(*blobNameToTensor->find(s.c_str()));
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setInt8Mode(false);
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

float doInference(IExecutionContext& context, float* inputData, float* prob, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    float ms = 0.0f;
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
	outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME),
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_CLS_SIZE * sizeof(float)));              // prob
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
    CHECK(cudaMemcpyAsync(prob, buffers[outputIndex], batchSize * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    cudaStreamDestroy(stream);
    return ms;
}

bool cmp(const std::pair<std::string, float> a, const std::pair<std::string, float> b)
{
    return a.second > b.second;
}

RES do_each_batch(unsigned int N, int beginIdx, IExecutionContext *context, const std::vector<std::string> classList, 
		 float *data, float *prob, PPM *ppms)
{
    memset(data, 0x0, N * INPUT_C * INPUT_H * INPUT_W * sizeof(float));
    memset(prob, 0x0, N * OUTPUT_CLS_SIZE * sizeof(float));
    memset(ppms, 0x0, N * sizeof(PPM));
    RES res = {0};
    std::vector<std::string> imageList;
    std::vector<std::string> clazList;
    for (unsigned int i = beginIdx; i < beginIdx + N; ++i)
    {
	imageList.push_back("/home/cd/TensorRT-4.0.1.6/data/resnet/ppms/" + std::to_string(i) + ".ppm");
	clazList.push_back(classList[i]);
    }
    assert(imageList.size() == N);
    assert(clazList.size() == N);
    for (unsigned int i = 0; i < N; ++i)
    	readPPMFile(imageList[i], ppms[i]);
    float pixelMean[3]{ 102.852869f, 115.518302f, 121.358954f }; // also in BGR order
    for (unsigned int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
    {
	for (unsigned int c = 0; c < INPUT_C; ++c)
	{
	    // the color image to input should be in BGR order
	    for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
		data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
	}
    }
    res.totalTime = doInference(*context, data, prob, N);
    int t1_success = 0, t5_success = 0;
    for (unsigned int i = 0; i < N; ++i)
    {
	std::vector<std::pair<std::string, float>> vec;
	for (int j = 0; j < OUTPUT_CLS_SIZE; ++j)
	    vec.push_back(make_pair(std::to_string(j), prob[j + i * OUTPUT_CLS_SIZE]));
	sort(vec.begin(), vec.end(), cmp);
	if (vec[0].first == clazList[i])
	    t1_success++;
	if ((vec[0].first == clazList[i]) || (vec[1].first == clazList[i]) || (vec[2].first == clazList[i]) || 
	    (vec[3].first == clazList[i]) || (vec[4].first == clazList[i]))
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
    IHostMemory *modelStream{ nullptr };
    const int N = 10;
    const int total_number = 3000;
    caffeToTRTModel("/home/cd/TensorRT-4.0.1.6/data/resnet/Resnet-50-deploy.prototxt", 
		    "/home/cd/TensorRT-4.0.1.6/data/resnet/Resnet-50-model.caffemodel", 
		    std::vector < std::string > { OUTPUT_BLOB_NAME }, 
		    N, &modelStream, DataType::kINT8);
    std::vector<std::string> classList;
    std::ifstream class_infile("/home/cd/TensorRT-4.0.1.6/data/resnet/ground-truth.txt");
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
    float* data = new float[N  *INPUT_C * INPUT_H * INPUT_W];
    float* prob = new float[N * OUTPUT_CLS_SIZE];
    PPM* ppms = new PPM[N];
	
    for (int i = 0; i < total_number/N; ++i)
    {
	std::cout << "do batch " << i < " ..." << std::endl;
	res = do_each_batch(N, i * N, context, classList, data, prob, ppms);
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
