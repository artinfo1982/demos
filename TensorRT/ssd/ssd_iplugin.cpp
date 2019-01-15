#include "ssd_iplugin.h"

/*
below layers must implement plugin in TensorRT-4.0.1.6
enum class PluginType : int
{
  kFASTERRCNN = 0,  //FasterRCNN fused plugin (RPN + ROI pooling)
  kNORMALIZE = 1,  //Normalize plugin
  kPERMUTE = 2, //Permute plugin
  kPRIORBOX = 3,  //PriorBox plugin
  kSSDDETECTIONOUTPUT = 4,  //SSD DetectionOutput plugin
  kConcat = 5,  //Concat plugin
  kPRELU = 6, //YOLO PReLU plugin
  kYOLOREORG = 7, //YOLO Reorg plugin
  kYOLOREGION = 8,  //YOLO Region plugin
  kANCHORGENERATOR = 9 //SSD Grid Anchor Generator
};
*/

nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
  assert(isPlugin(layerName));
  PriorBoxParameters prior_box_param;
  prior_box_param.numMinSize = 1;
  prior_box_param.numMaxSize = 1;
  prior_box_param.flip = true;
  prior_box_param.clip = false;
  prior_box_param.variance[0] = 0.1f;
  prior_box_param.variance[1] = 0.1f;
  prior_box_param.variance[2] = 0.2f;
  prior_box_param.variance[3] = 0.2f;
  prior_box_param.imgH = 0;
  prior_box_param.imgW = 0;
  prior_box_param.offset = 0.5f;
  DetectionOutputParameters detection_output_param;
  detection_output_param.shareLocation = true;
  detection_output_param.varianceEncodeedInTarget = false;
  detection_output_param.backgroundLabelId = 0;
  detection_output_param.numClasses = 21;
  detection_output_param.topK = 400;
  detection_output_param.keepTopK = 200;
  detection_output_param.confidenceThreshold = 0.1f;
  detection_output_param.nmsThreshold = 0.45f;
  detection_output_param.inputOrder[0] = 0;
  detection_output_param.inputOrder[1] = 1;
  detection_output_param.inputOrder[2] = 2;
  detection_output_param.confSigmoid = false;
  detection_output_param.isNormalized = true;
  Quadruple permute_param;
  permute_param.data[0] = 0;
  permute_param.data[1] = 2;
  permute_param.data[2] = 3;
  permute_param.data[3] = 1;
  //normalize layer
  if (!strcmp(layerName, "conv4_3_norm"))
  {
    assert(mNormalizeLayer == nullptr);
    bool acrossSpatial = false, channelShared = false;
    float eps = 0.0001;
    mNormalizeLayer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDNorm)
  }
}
